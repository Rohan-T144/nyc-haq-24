from itertools import product

from scipy.linalg import eigvals, eigh
from openfermion.utils import hermitian_conjugated as hc
from tangelo import SecondQuantizedMolecule
from tangelo.linq import Circuit, Gate, get_backend
from tangelo.toolboxes.ansatz_generator.ansatz_utils import (
    trotterize,
)
from tangelo.toolboxes.operators import QubitOperator, FermionOperator, count_qubits
from tangelo.toolboxes.qubit_mappings.mapping_transform import (
    fermion_to_qubit_mapping as f2q_mapping,
)
from tangelo.toolboxes.qubit_mappings.statevector_mapping import (
    vector_to_circuit,
)

from tangelo.algorithms.variational import VQESolver, BuiltInAnsatze, SA_VQESolver
from tangelo.algorithms.classical import CCSDSolver

import pennylane as qml
import pennylane
from pennylane import qchem
from pennylane import numpy as np


class HamiltonianSimulator:
    def __init__(self, mol: SecondQuantizedMolecule, mapping="jw"):
        """
        Constructor for HamiltonianSimulator.

        Parameters
        ----------
        mol : SecondQuantizedMolecule
            The molecule to simulate.
        mapping : str, optional
            The qubit mapping to use. Defaults to "jw".

        Returns
        -------
        None
        """
        ############################################################################
        # Hamiltonian simulation using Trotter-Suzuki and MRSQK                    #
        ############################################################################
        # Number of Krylov vectors
        self.n_krylov = 4

        # Qubit Mapping
        self.mapping = mapping
        # mapping = 'BK'
        # mapping = 'scBK'

        # Molecule
        self.mol = mol

        self.qu_op = f2q_mapping(
            mol.fermionic_hamiltonian,
            mapping,
            mol.n_active_sos,
            mol.n_active_electrons,
            up_then_down=False,
            spin=mol.spin,
        )
        self.backend = get_backend()

        self.c_q = count_qubits(self.qu_op)
        # print(f"Qubit count: {self.c_q+1}, control qubit: {self.c_q}")

        self.zeroone = QubitOperator(f"X{self.c_q}", 1) + QubitOperator(
            f"Y{self.c_q}", 1j
        )

        # Generate multiple controlled-reference states.
        self.reference_states: list[Circuit] = list()
        reference_vecs = [[1, 1, 0, 0], [1, 0, 0, 1]]
        for vec in reference_vecs:
            circ = vector_to_circuit(vec)
            gates = [
                Gate("C" + gate.name, target=gate.target, control=self.c_q)
                for gate in circ
            ]
            self.reference_states += [Circuit(gates)]

        ############################################################################
        # QChem simulation using Pennylane                                         #
        ############################################################################
        def init_qchem(self):
            elements = [e[0] for e in mol.xyz]
            coords = np.array([np.array(e[1]) for e in mol.xyz])
            self.qchem_mol = qchem.Molecule(elements, coords, charge=mol.q)
            # mol.n_electrons
            # electrons = 4
            self.qchem_hamiltonian, self.qchem_qubits = qchem.molecular_hamiltonian(
                self.qchem_mol
            )

            hf = qchem.hf_state(mol.n_electrons, self.qchem_qubits)
            singles, doubles = qchem.excitations(
                electrons=mol.n_electrons, orbitals=self.qchem_qubits
            )
            num_theta = len(singles) + len(doubles)

            self.device = qml.device("lightning.qubit", wires=self.qchem_qubits)

            def circuit_VQE(theta, wires):
                qml.AllSinglesDoubles(
                    weights=theta,
                    wires=wires,
                    hf_state=hf,
                    singles=singles,
                    doubles=doubles,
                )

            @qml.qnode(self.device, interface="autograd")
            def cost_fn(theta):
                circuit_VQE(theta, range(self.qchem_qubits))
                return qml.expval(self.qchem_hamiltonian)

            stepsize = 0.4
            max_iterations = 50
            opt = qml.GradientDescentOptimizer(stepsize=stepsize)
            theta = pennylane.numpy.zeros(num_theta, requires_grad=True)

            for n in range(max_iterations):
                theta, prev_energy = opt.step_and_cost(cost_fn, theta)

            # energy_VQE = cost_fn(theta)
            # theta_opt = theta

            @qml.qnode(self.device)
            def qdrift_t(time: float):
                # Prepare some state
                # qml.Hadamard(0)
                circuit_VQE(theta, range(self.qchem_qubits))
                # Evolve according to H
                qml.QDrift(self.qchem_hamiltonian, time=time, n=10, seed=10)

                return qml.expval(self.qchem_hamiltonian)

            self.qdrift_t = qdrift_t

        self.init_qchem = init_qchem

    def simulate_qdrift(self, t_end: float) -> float:
        """
        Simulates the time-evolution of the quantum chemistry hamiltonian up to time t_end.

        Args:
            t_end (float): The end time of the simulation.

        Returns:
            float: The expectation value (energy) of the quantum chemistry hamiltonian at t_end.
        """
        if not hasattr(self, "qdrift_t"):
            self.init_qchem(self)
        return self.qdrift_t(t_end)

    def simulate_mrsqk(self, t_end: float, n_steps: int = 1) -> np.array:
        """
        Simulates the time-evolution of the quantum chemistry hamiltonian up to time t_end
        using the MRSQK algorithm and Trotter-Suzuki decomposition.

        Args:
            t_end (float): The end time of the simulation.
            n_steps (int, optional): The number of steps to take in the Trotterized
                time-evolution. Defaults to 1.

        Returns:
            np.array: The expectation value (energy) of the quantum chemistry hamiltonian
                at t_end.
        """
        print(f"Qubit count: {self.c_q+1}, control qubit: {self.c_q}")
        c_trott = trotterize(
            self.qu_op,
            time=t_end,
            control=self.c_q,
            n_trotter_steps=n_steps,
            # n_trotter_steps=int(np.ceil(t_end/0.2))
        )

        # Calculate MRSQK
        sab = np.zeros((self.n_krylov, self.n_krylov), dtype=complex)
        # hab = np.zeros((self.n_krylov, self.n_krylov), dtype=complex)
        fhab = np.zeros((self.n_krylov, self.n_krylov), dtype=complex)

        resources = {}

        for a, b in product(range(self.n_krylov), range(self.n_krylov)):
            # Only need to do the upper triangle,
            # and the rest is the conjugate by symmetry
            if a < b:
                continue

            # Generate Ua and Ub unitaries
            ua = (
                self.reference_states[a % 2] + c_trott * (a // 2)
                if a > 1
                else self.reference_states[a % 2]
            )
            ub = (
                self.reference_states[b % 2] + c_trott * (b // 2)
                if b > 1
                else self.reference_states[b % 2]
            )

            # Build circuit from Figure 2 for off-diagonal overlap
            hab_circuit = (
                Circuit([Gate("H", self.c_q)])
                + ua
                + Circuit([Gate("X", self.c_q)])
                + ub
            )

            sab[a, b] = (
                self.backend.get_expectation_value(self.zeroone, hab_circuit) / 2
            )
            sab[b, a] = sab[a, b].conj()

            # Hamiltonian matrix element for f(H) = e^{-i H \tau}
            fhab[a, b] = (
                self.backend.get_expectation_value(
                    self.zeroone, hab_circuit + c_trott.inverse()
                )
                / 2
            )
            # hab_circuit.
            resources = {
                **resources,
                **hab_circuit.counts_n_qubit,
                "width": hab_circuit.width,
                "depth": hab_circuit.depth(),
            }

        print(f"Circuit resources:\n{resources}")

        # e, v = scipy.linalg.eigh(hab, sab)
        # print(f"The HV=SVE energies are {np.real_if_close(np.round(e, 3))}")
        e = eigvals(fhab, sab)
        # print(f"The f(H)V=SVf(E) energies are {np.arctan2(np.imag(e), np.real(e))/tau}")
        return np.arctan2(np.imag(e), np.real(e)) / t_end


class MoleculeSimulator:
    def __init__(self, mol: SecondQuantizedMolecule) -> None:
        """
        Initialize the MoleculeSimulator.

        Parameters
        ----------
        mol : SecondQuantizedMolecule
            The molecule to simulate.

        Returns
        -------
        None
        """
        self.mol = mol
        self.h_sim = HamiltonianSimulator(mol)

    def ground_state_vqe(self) -> float:
        """
        Compute the ground state energy of the molecule using VQE.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The ground state energy of the molecule.
        """
        vqe_solver = VQESolver({"molecule": self.mol, "ansatz": BuiltInAnsatze.UCCSD})
        vqe_solver.build()
        vqe_energy = vqe_solver.simulate()
        print(vqe_solver.get_resources())
        print(f"Ground state \t VQE energy = {vqe_energy}")

        return vqe_energy

    def ground_state_classical(self) -> float:
        """
        Compute the ground state energy of the molecule using classical CCSD.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The ground state energy of the molecule.
        """
        ccsd_solver = CCSDSolver(self.mol)
        return ccsd_solver.simulate()

    def excited_states_sa_vqe(self):
        """
        Compute the excited states of the molecule using Shift-Average VQE.

        This function implements the Shift-Average VQE algorithm as described in
        arXiv:1901.01234. It computes the first three excited states of the molecule.

        Parameters
        ----------
        None

        Returns
        -------
        np.array
            An array of the first three excited states of the molecule.
        """
        vqe_options = {
            "molecule": self.mol,
            "ref_states": [[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]],
            "weights": [1, 1, 1],
            "penalty_terms": {"S^2": [2, 0]},
            "qubit_mapping": "jw",
            "ansatz": BuiltInAnsatze.UpCCGSD,
        }
        vqe_solver = SA_VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()
        # for i, energy in enumerate(vqe_solver.state_energies):
        #     print(f"Singlet State {i} has energy {energy}")

        # Generate individual statevectors
        ref_svs = list()
        for circuit in vqe_solver.reference_circuits:
            _, sv = vqe_solver.backend.simulate(circuit, return_statevector=True)
            ref_svs.append(sv)

        # Generate Equation (2) using equation (4) and (5) of arXiv:1901.01234
        h_theta_theta = np.zeros((3, 3))
        for i, sv1 in enumerate(ref_svs):
            for j, sv2 in enumerate(ref_svs):
                if i != j:
                    sv_plus = (sv1 + sv2) / np.sqrt(2)
                    sv_minus = (sv1 - sv2) / np.sqrt(2)
                    exp_plus = vqe_solver.backend.get_expectation_value(
                        vqe_solver.qubit_hamiltonian,
                        vqe_solver.optimal_circuit,
                        initial_statevector=sv_plus,
                    )
                    exp_minus = vqe_solver.backend.get_expectation_value(
                        vqe_solver.qubit_hamiltonian,
                        vqe_solver.optimal_circuit,
                        initial_statevector=sv_minus,
                    )
                    h_theta_theta[i, j] = (exp_plus - exp_minus) / 2
                else:
                    h_theta_theta[i, j] = vqe_solver.state_energies[i]

        e, _ = np.linalg.eigh(h_theta_theta)
        # print(e)
        for i, energy in enumerate(e):
            print(f"Singlet State {i} \t MC-VQE energy = {energy}")

        return e

        # algorithm_resources["sa_vqe"] = vqe_solver.get_resources()

    def excited_states_vqe_deflation(self, k: int) -> np.array:
        """
        Compute the first k excited states (including the ground state)
        using the deflation technique.

        This method first performs a VQE optimization using the UCCSD ansatz,
        and then uses the resulting circuit as the first deflation circuit.
        It then performs a series of VQE optimizations using the UpCCGSD ansatz,
        each time adding the previous optimal circuit to the deflation circuits
        list. The final energies are returned as an array.

        Args:
            k (int): The number of excited states to compute.

        Returns:
            np.array: An array of length k containing the energies of the
            ground state and the first k-1 excited states.
        """
        vqe_solver = VQESolver({"molecule": self.mol, "ansatz": BuiltInAnsatze.UCCSD})
        vqe_solver.build()
        vqe_energy = vqe_solver.simulate()
        energies = [vqe_energy]
        print(vqe_solver.get_resources())
        print(f"Ground state \t VQE energy = {vqe_energy}")

        # Add initial VQE optimal circuit to the deflation circuits list
        deflation_circuits = [vqe_solver.optimal_circuit.copy()]

        # Calculate first and second excited states by adding optimal circuits to deflation_circuits
        for i in range(1, k):
            vqe_solver = VQESolver(
                {
                    "molecule": self.mol,
                    "ansatz": BuiltInAnsatze.UpCCGSD,
                    "deflation_circuits": deflation_circuits,
                    "deflation_coeff": 0.4,
                }
            )
            vqe_solver.build()
            vqe_energy = vqe_solver.simulate()
            print(vqe_solver.get_resources())
            print(f"Excited state #{i} \t VQE energy = {vqe_energy}")
            energies.append(vqe_energy)

        return np.array(energies)

    def excited_states_qse(self) -> np.array:
        """
        Compute the excited states of the molecule using Quantum Subspace Expansion (QSE).

        This function first performs a VQE optimization using the UCCSD ansatz.
        It then generates all single excitations as qubit operators and computes
        the F and S matrices. The eigenvalues of the generalized eigenvalue problem
        Hv = EvS are then computed and returned as an array.

        Returns:
            np.array: An array containing the energies of the excited states.
        """
        vqe_solver = VQESolver({"molecule": self.mol, "ansatz": BuiltInAnsatze.UCCSD})
        vqe_solver.build()
        vqe_solver.simulate()

        # Generate all single excitations as qubit operators
        op_list = list()
        for i in range(2):
            for j in range(i + 1, 2):
                op_list += [
                    f2q_mapping(FermionOperator(((2 * i, 1), (2 * j, 0))), "jw")
                ]  # spin-up transition
                op_list += [
                    f2q_mapping(FermionOperator(((2 * i + 1, 1), (2 * j + 1, 0))), "jw")
                ]  # spin-down transition
                op_list += [
                    f2q_mapping(FermionOperator(((2 * i + 1, 1), (2 * j, 0))), "jw")
                ]  # spin-up to spin-down
                op_list += [
                    f2q_mapping(FermionOperator(((2 * i, 1), (2 * j + 1, 0))), "jw")
                ]  # spin-down to spin-up

        # Compute F and S matrices.
        size_mat = len(op_list)
        h = np.zeros((size_mat, size_mat))
        s = np.zeros((size_mat, size_mat))
        state_circuit = vqe_solver.optimal_circuit
        for i, op1 in enumerate(op_list):
            for j, op2 in enumerate(op_list):
                h[i, j] = np.real(
                    vqe_solver.backend.get_expectation_value(
                        hc(op1) * vqe_solver.qubit_hamiltonian * op2, state_circuit
                    )
                )
                s[i, j] = np.real(
                    vqe_solver.backend.get_expectation_value(
                        hc(op1) * op2, state_circuit
                    )
                )

        e, v = eigh(h, s)
        return e

    def hamiltonian_t_mrsqk(self, t: float) -> np.array:
        """
        Compute the Hamiltonian matrix representation at time t using the
        Multi Reference Symmetric Qubitization Krylov (MRSQK) algorithm.

        Parameters
        ----------
        t : float
            The time at which to compute the Hamiltonian matrix representation.

        Returns
        -------
        np.array
            A 2D array representing the Hamiltonian matrix at time t.
        """
        return self.h_sim.simulate_mrsqk(t)

    def hamiltonian_t_qdrift(self, t: float) -> float:
        """
        Compute the expectation value of the Hamiltonian at time t using the
        Quantum Drift (QDrift) algorithm.

        Parameters
        ----------
        t : float
            The time at which to compute the expectation value of the Hamiltonian.

        Returns
        -------
        float
            The expectation value of the Hamiltonian at time t.
        """
        return self.h_sim.simulate_qdrift(t)

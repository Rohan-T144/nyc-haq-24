from tangelo import SecondQuantizedMolecule
from tangelo.algorithms import VQESolver, CCSDSolver
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import signal
from contextlib import contextmanager


def print_molecule_properties(name, molecule):
    print(f"=== {name} Properties ===")
    print(f"Charge (q): {molecule.q}")
    print(f"Spin (2S): {molecule.spin}")
    print(f"Number of Electrons: {molecule.n_electrons}")
    print(f"Number of Active Electrons: {molecule.n_active_electrons}")
    print(f"Basis Set: {molecule.basis}")
    print(f"Frozen Orbitals: {molecule.frozen_orbitals}")
    print("--------------------------\n")


def print_orbital_energies(molecule):
    orbital_energies = molecule.mo_energies
    occupations = molecule.mo_occ

    orbital_info = []
    for idx, (energy, occ) in enumerate(zip(orbital_energies, occupations)):
        orbital_info.append((idx, energy, occ))

    # Display orbital information
    print(f"Number of active electrons: {molecule.n_active_electrons}\n")
    print("Orbital Index | Occupation Number | Orbital Energy (Hartree)")
    print("------------------------------------------------------------")
    for idx, energy, occ in orbital_info:
        print(f"{idx:<13} {occ:<18} {energy}")

def simulate_classically(second_quantized_mol):
    ccsd_solver = CCSDSolver(second_quantized_mol)
    energy_ccsd = ccsd_solver.simulate()
    return energy_ccsd

def vqe_simulate(second_quantized_mol):
    vqe_options = {"molecule": second_quantized_mol}
    vqe_solver = VQESolver(vqe_options)
    vqe_solver.build()
    return vqe_solver.simulate()

def vqe_resources(second_quantized_mol):
    vqe_options = {"molecule": second_quantized_mol}
    vqe_solver = VQESolver(vqe_options)
    build = vqe_solver.build()
    return build, vqe_solver.get_resources()


def KMeans1D_DP(data, K):
    n = len(data)
    # Sort data and keep track of original indices
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    # Precompute cumulative sums and cumulative squared sums
    data_cumsum = np.zeros(n + 1)
    data_sq_cumsum = np.zeros(n + 1)
    for i in range(1, n + 1):
        data_cumsum[i] = data_cumsum[i - 1] + sorted_data[i - 1]
        data_sq_cumsum[i] = data_sq_cumsum[i - 1] + sorted_data[i - 1] ** 2

    # Initialize the cost and segmentation arrays
    cost = np.full((n + 1, K + 1), np.inf)
    backtrack = np.zeros((n + 1, K + 1), dtype=int)

    cost[0][0] = 0  # Base case

    # Dynamic programming to compute optimal cost
    for i in range(1, n + 1):
        for k in range(1, min(i, K) + 1):
            for j in range(k - 1, i):
                count = i - j
                sum_x = data_cumsum[i] - data_cumsum[j]
                sum_x_sq = data_sq_cumsum[i] - data_sq_cumsum[j]
                mean = sum_x / count
                ssd = sum_x_sq - 2 * mean * sum_x + count * mean**2
                total_cost = cost[j][k - 1] + ssd
                if total_cost < cost[i][k]:
                    cost[i][k] = total_cost
                    backtrack[i][k] = j

    # Backtracking to assign labels
    labels = np.zeros(n, dtype=int)
    k = K
    i = n
    while k > 0:
        j = backtrack[i][k]
        labels[j:i] = k - 1
        i = j
        k -= 1

    # Map labels back to original data order
    original_labels = np.zeros(n, dtype=int)
    original_labels[sorted_indices] = labels

    return original_labels


def generate_freeze_lists(molecule, k_core, k_virtual):
    orbital_energies = molecule.mo_energies
    occupations = molecule.mo_occ

    orbital_info = []
    for idx, (energy, occ) in enumerate(zip(orbital_energies, occupations)):
        orbital_info.append((idx, energy, occ))

    print(f"Number of active electrons: {molecule.n_active_electrons}")

    # Collect core orbitals (occupation == 2.0)
    core_indices = []
    core_energies = []
    for idx, energy, occ in orbital_info:
        if occ == 2.0:
            core_indices.append(idx)
            core_energies.append(energy)

    core_indices = np.array(core_indices)
    core_energies = np.array(core_energies)

    # Cluster core orbitals using dynamic programming
    if len(core_indices) == 0:
        print("No core orbitals to cluster.")
        core_clusters = {}
        core_cluster_order = []
    else:
        k_core = min(k_core, len(core_indices))
        core_labels = KMeans1D_DP(core_energies, k_core)
        # Group core orbitals by clusters
        core_clusters = {}
        for label in range(k_core):
            cluster_indices = core_indices[core_labels == label]
            core_clusters[label] = cluster_indices
        core_cluster_order = range(k_core)
        # Print cluster info
        print(f"\nClustered core orbitals into {k_core} groups:")
        for label in core_cluster_order:
            indices = core_clusters[label]
            energies = [orbital_energies[i] for i in indices]
            print(f"Core Cluster {label+1}: Orbitals {indices}, Energies {energies}")

    # Collect virtual orbitals (occupation == 0.0)
    virtual_indices = []
    virtual_energies = []
    for idx, energy, occ in orbital_info:
        if occ == 0.0:
            virtual_indices.append(idx)
            virtual_energies.append(energy)

    virtual_indices = np.array(virtual_indices)
    virtual_energies = np.array(virtual_energies)

    # Cluster virtual orbitals using dynamic programming
    if len(virtual_indices) == 0:
        print("No virtual orbitals to cluster.")
        virtual_clusters = {}
        virtual_cluster_order = []
    else:
        k_virtual = min(k_virtual, len(virtual_indices))
        virtual_labels = KMeans1D_DP(virtual_energies, k_virtual)
        # Group virtual orbitals by clusters
        virtual_clusters = {}
        for label in range(k_virtual):
            cluster_indices = virtual_indices[virtual_labels == label]
            virtual_clusters[label] = cluster_indices
        virtual_cluster_order = range(k_virtual)
        # Print cluster info
        print(f"\nClustered virtual orbitals into {k_virtual} groups:")
        for label in virtual_cluster_order:
            indices = virtual_clusters[label]
            energies = [orbital_energies[i] for i in indices]
            print(f"Virtual Cluster {label+1}: Orbitals {indices}, Energies {energies}")

    # Generate combinations according to the specified logic
    freeze_lists = []
    num_core_clusters = len(core_cluster_order)
    num_virtual_clusters = len(virtual_cluster_order)

    for c in range(1, num_core_clusters + 1):
        # Core clusters to include
        core_clusters_to_include = core_cluster_order[:c]
        core_orbitals_to_freeze = []
        for label in core_clusters_to_include:
            core_orbitals_to_freeze.extend(core_clusters[label])
        for v in range(0, num_virtual_clusters + 1):
            # Virtual clusters to include
            virtual_clusters_to_include = virtual_cluster_order[-v:] if v > 0 else []
            virtual_orbitals_to_freeze = []
            for label in virtual_clusters_to_include:
                virtual_orbitals_to_freeze.extend(virtual_clusters[label])
            freeze_list = core_orbitals_to_freeze + virtual_orbitals_to_freeze
            freeze_lists.append(freeze_list)
            print(
                f"\nGenerated freeze list with core clusters {core_clusters_to_include} and virtual clusters {virtual_clusters_to_include}"
            )
            print(f"Freeze list orbitals: {freeze_list}")

    return freeze_lists


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, signal_handler)
    # Set the alarm
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def run_with_timeout(func, args=(), kwargs=None, timeout=10):
    if kwargs is None:
        kwargs = {}
    try:
        with time_limit(timeout):
            return func(*args, **kwargs)
    except TimeoutException:
        print(f"Function '{func.__name__}' timed out after {timeout} seconds.")
        return -1
    except Exception as e:
        print(f"Function '{func.__name__}' raised an exception: {e}")
        return -2


def create_resources_dict(mol, charge, mol_spin, freeze_lists, timeout_duration=10):
    # Initialize the dictionary to store indexed entries
    frozen_molecules_dict = {}
    index = 0  # Start index at 0

    # Sort freeze_lists by length in descending order
    freeze_list_sorted = sorted(freeze_lists, key=len, reverse=True)

    for idx, freeze_list in enumerate(freeze_list_sorted, start=1):
        # Convert all elements in freeze_list to integers
        int_freeze_list = [int(x) for x in freeze_list]

        try:
            # Create a new SecondQuantizedMolecule with frozen orbitals
            frozen_orbs_mol = SecondQuantizedMolecule(
                mol,
                q=charge,
                spin=mol_spin,
                basis="LANL2DZ",
                frozen_orbitals=int_freeze_list,
            )

            print(
                f"\nCreating frozen molecule for Freeze List {idx}: {int_freeze_list}"
            )

            # Calculate resources with timeout
            result = run_with_timeout(
                vqe_resources, args=(frozen_orbs_mol,), timeout=timeout_duration
            )

            if result == -1:
                print(
                    f"Stopping at Freeze List {int_freeze_list} due to timeout or error in resource calculation."
                )
                break
            elif result == -2:
                print(
                    f"An error occurred while processing Freeze List: {int_freeze_list}"
                )
            else:
                # Unpack the result correctly
                build, resources = result

                # Ensure no None values are included
                if resources is not None:
                    # Store the index as key and a dict of molecule, freeze_list, and resources as value
                    frozen_molecules_dict[index] = {
                        "molecule": frozen_orbs_mol,
                        "freeze_list": int_freeze_list,
                        "resources": resources,
                    }
                    index += 1  # Increment the index
                else:
                    print(
                        f"Resources for Freeze List {int_freeze_list} are None. Skipping."
                    )

        except TypeError as e:
            print(f"TypeError for Freeze List {idx}: {e}")
        except Exception as e:
            print(f"Unexpected error for Freeze List {idx}: {e}")

    # Print the complete dictionary after processing
    print("\n--- Frozen Molecules Dictionary ---")
    for idx, res_dict in frozen_molecules_dict.items():
        print(f"Index {idx}: {res_dict}")

    return frozen_molecules_dict


def draw_resources(frozen_molecules_dict):
    # -----------------------------
    # Step 1: Prepare  Data
    # -----------------------------

    # Initialize lists to collect data
    num_frozen_orbitals_list = []
    circuit_widths_list = []
    circuit_depths_list = []
    vqe_variational_parameters_list = []

    for idx, data in frozen_molecules_dict.items():
        resources = data['resources']
        freeze_list = data['freeze_list']
        num_frozen_orbitals_list.append(len(freeze_list))
        circuit_widths_list.append(resources['circuit_width'])
        circuit_depths_list.append(resources['circuit_depth'])
        vqe_variational_parameters_list.append(resources['vqe_variational_parameters'])

    # Convert lists to NumPy arrays
    num_frozen_orbitals = np.array(num_frozen_orbitals_list)
    circuit_widths = np.array(circuit_widths_list)
    circuit_depths = np.array(circuit_depths_list)
    vqe_variational_parameters = np.array(vqe_variational_parameters_list)

    # -----------------------------
    # Step 2: Apply Jitter
    # -----------------------------

    jitter_x = 0.2  # Reduced jitter magnitude for cleaner plot
    jitter_y = 0.2

    np.random.seed(42)  # For reproducibility

    # Apply jitter to x and y coordinates
    num_frozen_orbitals_jittered = num_frozen_orbitals + np.random.uniform(-jitter_x, jitter_x, size=num_frozen_orbitals.shape)
    circuit_widths_jittered = circuit_widths + np.random.uniform(-jitter_y, jitter_y, size=circuit_widths.shape)

    # -----------------------------
    # Step 3: Create Scatter Plot
    # -----------------------------

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        num_frozen_orbitals_jittered,
        circuit_widths_jittered,
        c=circuit_depths,
        cmap='RdYlGn_r',  # Reversed diverging colormap: green (low) to red (high)
        s=100,
        alpha=0.7,
        edgecolor='k'  # Black edge for better marker distinction
    )

    # -----------------------------
    # Step 4: Add Colorbar
    # -----------------------------

    cbar = plt.colorbar(scatter)
    cbar.set_label('Circuit Depth', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # -----------------------------
    # Step 5: Add Labels and Title
    # -----------------------------

    plt.xlabel('Number of Frozen Orbitals', fontsize=14)
    plt.ylabel('Circuit Width', fontsize=14)
    plt.title('Circuit Width vs. Number of Frozen Orbitals\nColored by Circuit Depth', fontsize=16)

    # -----------------------------
    # Step 6: Annotate Each Point
    # -----------------------------

    texts = []

    # Iterate over the jittered data and annotate
    for x, y, param in zip(num_frozen_orbitals_jittered, circuit_widths_jittered, vqe_variational_parameters):
        # Validate that param is a number and not NaN
        if param is not None and not (isinstance(param, float) and np.isnan(param)):
            texts.append(
                plt.text(
                    x, y,
                    f'{param}',
                    fontsize=9,
                    ha='center',
                    va='center',
                    color='black',
                    alpha=0.9,
                    bbox={'facecolor': 'white', 'alpha': 0.6, 'edgecolor': 'none', 'boxstyle': 'round,pad=0.2'}
                )
            )
        else:
            print(f"Invalid VQE Variational Parameter at ({x}, {y}): {param}")

    # -----------------------------
    # Step 7: Adjust Annotations
    # -----------------------------

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        only_move={'points': 'y', 'texts': 'y'},
        force_points=0.1,
        force_text=0.1,
        expand_points=(1.1, 1.2),
        expand_text=(1.1, 1.2),
        lim=50
    )

    # -----------------------------
    # Step 8: Final Touches
    # -----------------------------

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def run_vqe_simulations(frozen_molecules_dict, timeout_duration=10):
    """
    Takes the dictionary from create_resources_dict, creates a list of (circuit width, energy) tuples,
    sorts it in ascending order of circuit width, and runs VQE simulations with timeout.
    Stops when a timeout is encountered.
    After the simulations, runs a classical simulation on the last successfully simulated molecule and prints the energy.
    Returns a list of (circuit_width, energy) tuples from VQE simulations.
    """
    # Create a list to hold (circuit width, molecule object)
    circuit_molecule_list = []
    # List to store the VQE results
    energies_vs_circuit_width = []
    # Variables to keep track of the last successful molecule
    last_successful_molecule = None
    last_circuit_width = None

    # Extract circuit width and molecule from the dictionary
    for idx, res_dict in frozen_molecules_dict.items():
        resources = res_dict['resources']
        molecule = res_dict['molecule']
        # Get circuit width from resources
        circuit_width = resources.get('circuit_width', None)  # Adjust the key based on your resources structure
        if circuit_width is not None:
            circuit_molecule_list.append((circuit_width, molecule))
        else:
            print(f"No circuit width found for index {idx}. Skipping.")

    # Sort the list by circuit width in ascending order
    circuit_molecule_list.sort(key=lambda x: x[0])

    # Run VQE simulations with timeout, starting from the smallest circuit width
    for circuit_width, molecule in circuit_molecule_list:
        print(f"Running VQE simulation for circuit width {circuit_width}")
        result = run_with_timeout(vqe_simulate, args=(molecule,), timeout=timeout_duration)
        if result == -1:
            print(f"Stopping at circuit width {circuit_width} due to timeout.")
            break
        elif result == -2:
            print(f"An error occurred during VQE simulation for circuit width {circuit_width}")
            continue  # Skip to the next molecule
        else:
            # Simulation succeeded
            energy = result  # Assuming vqe_simulate returns the energy
            print(f"VQE simulation successful for circuit width {circuit_width}. Energy: {energy}")
            energies_vs_circuit_width.append((circuit_width, energy))
            # Update the last successful molecule
            last_successful_molecule = molecule
            last_circuit_width = circuit_width

    # After the VQE simulations, run classical simulation on the last successful molecule
    if last_successful_molecule is not None:
        print(f"\nRunning classical simulation for the last successfully simulated molecule with circuit width {last_circuit_width}")
        classical_energy = simulate_classically(last_successful_molecule)
        print(f"Classical energy for circuit width {last_circuit_width}: {classical_energy}")
    else:
        print("No successful VQE simulations were performed before timeout.")

    print("\nVQE Energies vs. Circuit Width:")
    for cw, energy in energies_vs_circuit_width:
        print(f"Circuit Width: {cw}, VQE Energy: {energy}")

    return energies_vs_circuit_width


def plot_energies_vs_circuit_width(energies_vs_circuit_width):
    """
    Plots energies vs. circuit width as scatter points.
    Handles closely spaced energy values by appropriately scaling the y-axis.
    
    Parameters:
        energies_vs_circuit_width (list of tuples): List containing (circuit_width, energy) tuples.
    """
    if not energies_vs_circuit_width:
        print("No data to plot.")
        return
    
    # Unpack circuit widths and energies
    circuit_widths, energies = zip(*energies_vs_circuit_width)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with larger markers
    plt.scatter(
        circuit_widths,
        energies,
        color='blue',
        alpha=0.7,
        edgecolors='w',
        s=200,  # Increased marker size from 100 to 200
        linewidth=1.5,
        marker='o'
    )
    
    # Alternate annotation positions to avoid overlap
    for idx, (cw, energy) in enumerate(energies_vs_circuit_width):
        offset = 10 if idx % 2 == 0 else -15  # Alternate between above and below the marker
        plt.annotate(
            f"{energy:.12f}",
            (cw, energy),
            textcoords="offset points",
            xytext=(0, offset),
            ha='center',
            fontsize=8,
            fontweight='bold',
            color='darkred'
        )
    
    # Labeling
    plt.xlabel('Circuit Width', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.title('VQE Energies vs. Circuit Width', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Adjust y-axis limits for better visibility
    energy_min = min(energies)
    energy_max = max(energies)
    energy_range = energy_max - energy_min
    if energy_range == 0:
        # All energies are the same, set arbitrary range
        plt.ylim(energy_min - 1e-6, energy_max + 1e-6)
    else:
        plt.ylim(energy_min - 0.05 * energy_range, energy_max + 0.05 * energy_range)
    
    # Customize y-axis ticks for better readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.12f}'))
    # Alternatively, use scientific notation if values are very close
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6e'))
    
    plt.tight_layout()
    plt.show()
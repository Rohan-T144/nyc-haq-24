# Quantum Chemistry Simulation Project

## Overview

This project implements various quantum and classical algorithms for simulating the electronic structure and dynamics of the lithium hydride (LiH) molecule, and other toxic molecules like Cr(VI). It showcases modern techniques in quantum chemistry and quantum computing, providing a toolkit for molecular simulations.

See [`report.ipynb`](report.ipynb) for an overall explanation and demonstration of the project.

## Features

- Ground state energy calculations using:
  - Variational Quantum Eigensolver (VQE)
  - Classical Coupled Cluster Singles and Doubles (CCSD)
- Excited state calculations using:
  - Shift-Average Variational Quantum Eigensolver (SA-VQE)
  - VQE with Deflation
  - Quantum Subspace Expansion (QSE)
- Time evolution simulations using:
  - Multi-Reference Symmetric Qubitization Krylov (MRSQK)
    - Trotter-Suzuki decomposition
  - qDRIFT (using pennylane)
- Comparison of quantum and classical methods
- Visualization of results

## Requirements

- Python 3.8+
- Tangelo (Quantum chemistry package)
- Pennylane
- NumPy
- Matplotlib
- SciPy

You can install the required packages using:

```
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── qchem_algorithms.py
├── report.ipynb
├── qchem_opt.py
├── optimal_resources.ipynb
├── experiments/
└── figs/

```
<!-- │   ├── overview.md
│   ├── theoretical_background.md
│   ├── implementation_details.md
│   ├── results_and_analysis.md
│   └── conclusions.md
└── data/
    └── results/ -->

- `report.ipynb`: Jupyter notebook demonstrating the usage of algorithms and visualizing results.
- `qchem_algorithms.py`: Contains implementations of all quantum and classical algorithms used in the project.
- `optimal_resources.ipynb`: Notebook explaining our implementation of our meta-optimization algorithm, with examples on NO3 and Cr(III)
- `qchem_opt.py`: Contains implementations of our meta-optimization algorithm and related plotting.
- `experiments/`: Directory contains all the WIP and experimental code we created during the hackathon.
- `figs/`: Contains some figures generated as results from our simulations.

<!-- 
## Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/quantum-chemistry-simulation.git
   cd quantum-chemistry-simulation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```
   jupyter notebook quantum_chemistry_notebook.ipynb
   ```

4. Follow the instructions in the notebook to run simulations and visualize results. -->

<!-- ## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes. -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the Tangelo quantum chemistry package.
- NYC HAQ for their support throughout the process, organizing the hackathon, and proving many tools and lectures.
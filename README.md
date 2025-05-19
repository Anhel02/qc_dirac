# Quantum Simulation of the Dirac Equation

This repository contains Python code developed for a Final Degree Project in Physics on the Universidad Europea de Valencia, focused on simulating the Dirac equation using quantum computing techniques. The simulation estimates the ground state energy of the hydrogen atom via a parametrization of the Dirac Hamiltonian implemented on a quantum circuit.

## Overview

This project explores the use of quantum computing to simulate the Dirac equation, a cornerstone of relativistic quantum mechanics. The methodology involves encoding the Dirac Hamiltonian in a quantum circuit and estimating the ground state energy of the hydrogen atom using variational and time-evolution techniques.

The implementation relies on **Pennylane**, a quantum computing framework that facilitates hybrid quantum-classical computation.

## Features

- Initialization of the 1s orbital of the hydrogen atom on a 3D lattice
- Parameterized time evolution based on Trotterization and Quantum Walk techniques
- Implementation of custom potential, mass, and translation operators
- Quantum phase estimation to infer ground state energy
- Support for multiple spatial resolutions (`d = 2, 3, 4, 5`)

## Requirements

- Python ≥ 3.8  
- Pennylane  
- NumPy  
- Matplotlib *(optional, for visualizations)*
- **(Optional for GPU acceleration)**:
  - `jax`, `jaxlib` (with CUDA support)
  - `pennylane-lightning[gpu]`
  - Compatible GPU with CUDA ≥ 12.0 and GPU ≥ SM 7.0 (Volta)
  - CUDA libraries including `cusolver`

## How to Run

Run the script to compute the estimated energy values for different spatial resolutions (determined by d) and evolution steps:

```bash
python dirac.py
```

Output files (`d_2.txt`, `d_3.txt`, etc.) will be generated, containing the estimated energy values for each spatial resolution and step count.

If using GPU acceleration, uncomment the corresponding lines in dirac.py where the lightning.gpu device and jax interface are defined.

## References

This implementation is inspired by the following research paper:

Fillion-Gourdeau, F., MacLean, S., & Laflamme, R. (2017).  
*Algorithm for the solution of the Dirac equation on digital quantum computers.*  
Physical Review A, 95(4), 042343.  
[DOI: 10.1103/PhysRevA.95.042343](http://dx.doi.org/10.1103/PhysRevA.95.042343)

## Acknowledgments

Special thanks to the supervisor Ezequiel Valero Lafuente for his guidance throughout the project.

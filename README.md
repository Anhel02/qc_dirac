# Quantum Simulation of the Dirac Equation

This repository contains Python code developed for a Final Degree Project in Physics on the Universidad Europea de Valencia, focused on simulating the Dirac equation using quantum computing techniques. The simulation estimates the ground state energy of the hydrogen atom via a parametrization of the Dirac Hamiltonian implemented on a quantum circuit.

## Overview

This repository implements a quantum-circuit model for simulating the Dirac equation and using it as a proof-of-concept to estimate the ground-state energy of the hydrogen atom.

The approach discretises space on a cubic lattice and factors the Dirac time-evolution into a product of unitary operators that can be compiled into a gate-based circuit. In particular, the algorithm combines:

- **Lie–Trotter (Trotter–Suzuki) decomposition** to approximate time evolution with non-commuting terms
- **Operator splitting** to separate translation, mass and potential contributions
- A **quantum walk** construction for the conditional translation step

As an application, the circuit is initialised in a discretised hydrogen 1s orbital and evolved for a total time. Assuming the initial wavefunction is an eigenstate, the energy is extracted from the global phase.

Implementation is written in Python using PennyLane.

## Features

- Initialization of the 1s orbital of the hydrogen atom on a 3D lattice
- Parameterized time evolution based on Trotterization and Quantum Walk techniques
- Implementation of custom potential, mass, and translation operators
- Quantum phase estimation to infer ground state energy
- Support for multiple spatial resolutions (`d = 2, 3, 4, 5`)
- GPU acceleration.

## Requirements

- Python ≥ 3.8  
- Pennylane  
- NumPy  
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

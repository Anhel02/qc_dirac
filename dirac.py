# =============================================================
# Quantum Simulation of the Dirac Equation
# -------------------------------------------------------------
# Author: Ángel Gómez González
# Institution: Universidad Europea de Valencia
# Project: Bachelor's Thesis in Physics
# Academic Year: 2024–2025
# Supervisor: Ezequiel Valero Lafuente
#
# Description:
# This script simulates the time evolution governed by the Dirac
# equation using quantum computing techniques. It estimates the
# ground state energy of the hydrogen atom by encoding the Dirac
# Hamiltonian in a quantum circuit implemented with Pennylane.
#
# Reference:
# Fillion-Gourdeau, F., MacLean, S., & Laflamme, R. (2017).
# "Algorithm for the solution of the Dirac equation on digital
# quantum computers", Phys. Rev. A 95, 042343.
# DOI: https://doi.org/10.1103/PhysRevA.95.042343
# =============================================================

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

def evolution(d, steps):

    ##### Initial Parameters #####
    N=2**d
    L=10
    eps=1e-6
    dt=0.001
    m=1
    c=1

    wires = list(range(3 * d + 2))
    dev = qml.device('lightning.qubit', wires=wires)
    # If GPU >= SM 7.0 (Volta) + Cuda 12.9 + Jax  :
    # dev = qml.device('lightning.gpu', wires=wires)

    ##### Estado 1s #####

    def hydrogen_1s_state_vector(L, eps, N):
        coords = np.linspace(-L, L, N)
        psi = np.zeros((N, N, N))

        for i, x in enumerate(coords):
            for j, y in enumerate(coords):
                for k, z in enumerate(coords):
                    r = np.sqrt(x**2 + y**2 + z**2 + eps)
                    psi[i, j, k] = np.exp(-r)
        norm_sq = np.sum(np.abs(psi)**2)
        psi /= np.sqrt(norm_sq)
        return psi.flatten()

    ##### Operators #####

    def QV(dt, L, eps, wires, dims=3):
        n = len(wires)
        bits_per_dim = n // dims
        N = 2 ** bits_per_dim
        total = N ** dims
        l = 2 * L / (N - 1)
        diag = np.zeros(total, dtype=complex)

        for idx in range(total):
            coords = []
            tmp = idx
            for _ in range(dims):
                digit = tmp % N
                coords.append(digit)
                tmp //= N

            r_squared = 0
            for digit in coords:
                x = (digit - (N - 1) / 2) * l
                r_squared += x**2

            V_val = -1.0 / np.sqrt(r_squared + eps)
            diag[idx] = np.exp(-1j * V_val * dt)

        U = np.diag(diag)
        return qml.QubitUnitary(U, wires=wires)


    def controlled_decrement(wires):
        ctrl_wire = wires[0]
        target_wires = wires[1:]
        n = len(target_wires)

        inverted = set()

        for i in range(n):
            ctrl = [ctrl_wire] + target_wires[i+1:]
            to_invert = ctrl[1:]

            for wire in to_invert:
                if wire not in inverted:
                    qml.X(wires=wire)
                    inverted.add(wire)

            if i == n - 1:
                qml.CNOT(wires=[ctrl_wire, target_wires[i]])
            else:
                qml.MultiControlledX(wires=ctrl + [target_wires[i]])

            future_controls = set()
            for j in range(i + 1, n):
                future_controls.update(target_wires[j+1:])

            to_uninvert = inverted - future_controls

            for wire in to_uninvert:
                qml.X(wires=wire)
                inverted.remove(wire)


    def controlled_increment(wires):
        ctrl_wire = wires[0]
        target_wires = wires[1:]
        n = len(target_wires)
        qml.X(wires=ctrl_wire)
        for i in range(n):
            ctrl = [ctrl_wire] + target_wires[i+1:]
            if i == n - 1:
                qml.CNOT(wires=[ctrl_wire, target_wires[i]])
            else:
                qml.MultiControlledX(wires=ctrl + [target_wires[i]])
        qml.X(wires=ctrl_wire)

    ##### Dirac Circuit #####

    @qml.qnode(dev)
    # If 'lightning.gpu':
    #@qml.qnode(dev, interface='jax')
    def qc():
        for _ in range(steps):
            for gate, a in zip([qml.CNOT, qml.CY, qml.CZ], [0, d, 2 * d]):
                # Sa
                gate(wires=[wires[0], wires[1]])
                qml.Hadamard(wires=wires[0])
                gate(wires=[wires[0], wires[1]])

                qwires = [wires[0]] + wires[2 + a:2 + a + d]
                controlled_increment(qwires) # I
                controlled_decrement(qwires) # D

                # Sa
                gate(wires=[wires[0], wires[1]])
                qml.Hadamard(wires=wires[0])
                gate(wires=[wires[0], wires[1]])

            qml.RZ(2 * m * c**2 * dt, wires=[0]) # Qm
            QV(dt, L, eps, wires=range(2, 2 + 3 * d)) # Qv

        return qml.state()


    ##### Initialization #####

    @qml.qnode(dev)
    # If 'lightning.gpu':
    #@qml.qnode(dev, interface='jax')
    def prepare_initial_state():
        qml.BasisState(pnp.array([0, 0]), wires=[0, 1])
        psi = hydrogen_1s_state_vector(N, eps, N=2**d,)
        qml.StatePrep(psi, wires=range(2, 2 + 3 * d))
        return qml.state()


    ##### Energy Estimation #####

    T = steps * dt
    psi0 = prepare_initial_state()
    psiT = qc()
    overlap = np.vdot(psi0, psiT)
    phase = np.angle(overlap)
    energy = phase / T
    return energy


if __name__ == '__main__':

    ##### Get Data #####

    for d in [2,3,4,5]:
        filename = f'd_{d}.txt'
        with open(filename, 'a') as file:
            print(f'=== d = {d} ===')
            for steps in range(1, 10):
                energy = evolution(d=d, steps=steps)
                print(f'd={d}, steps={steps}: {energy}')
                file.write(f'{energy}\n')
                file.flush()

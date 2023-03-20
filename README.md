# Dynamic Programming on a Quantum Annealer

This repository contains a Python implementation of the algorithms introduced in Fernández-Villaverde and Hull (2022). It includes classical, classical combinatorial, hybrid, and quantum algorithms for solving dynamic programming problems in economics, and applies those algorithms to solve the Real Business Cycle (RBC) model. More generally, the novel use of reverse and inhomogenous annealing in Fernández-Villaverde and Hull (2022) can be applied to solve iterative algorithms on a quantum annealer without hybridizing the problem.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Solutions](#solutions)
   - [Classical Solution](#classical-solution)
   - [Classical Combinatorial Solution](#classical-combinatorial-solution)
   - [Hybrid Solution](#hybrid-solution)
   - [Quantum Solution](#quantum-solution)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/username/rbc-model-solver.git
cd rbc-model-solver
pip install -r requirements.txt
```

## Usage

Each solution is implemented as a separate Python script. The `hybrid_solution.py` and `quantum_solution.py` scripts require access to a D-Wave quantum annealer. Run the desired solution script as follows:

```bash
python classical_solution.py
```

## Solutions

### Classical Solution
The classical solution (`classical_solution.py`) uses parametric policy iteration (PPI) to find the optimal policy and valuation function parameters. It is implemented as an iterative algorithm, following Benitez-Silva et al. (2000).

### Classical Combinatorial Solution
The classical combinatorial solution (`classical_combinatorial_solution.py`) modifies the PPI solution by reframing the policy valuation step as a combinatorial optimization problem.

### Hybrid Solution
The hybrid solution (`hybrid_solution.py`) combines both classical and quantum components. It employs a quantum annealer to solve the policy valuation step, while the policy improvement step is computed classically.

### Quantum Solution
The quantum solutions (`quantum_solution.py`) use a quantum annealer to solve the RBC model with two different algorithms. Both algorithms encode the problem as a QUBO. The optimal policy and value function parameters are found through an iterative process across anneals or within an anneal that relies on the use of reverse and inhomogenous annealing.

## References

Fernández-Villaverde, Jesús, and Isaiah Hull. "Dynamic Programming on a Quantum Annealer: Solving the RBC Model." (2022).
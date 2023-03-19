"""
This module contains the `ClassicalCombinatorialSolution` class, which provides
a classical combinatorial solution to the Real Business Cycle (RBC) that builds
on Parametric Policy Iteration (PPI). The class provides methods to implement the 
policy improvement and policy valuation steps, and a method to solve the problem 
iteratively.

"""


import pickle
import time
import dimod
import numpy as np
import pandas as pd
from constants.classical import ALPHA, BETA, X1_TRUE, X2_TRUE, X3_TRUE
from constants.classical_combinatorial import N2, N3, S2, S3, U2, U3
from utilities.utils import Binary


class ClassicalCombinatorialSolution:
    """
    A class that implements a classical combinatorial algorithm for solving a combinatorial 
    optimization problem.

    Parameters:
    - x1_init (float), optional: the initial value for x1.
    - x2_init (float), optional: the initial value for x2.
    - x3_init (float), optional: the initial value for x3.

    Attributes:
    - qubo (dict): the QUBO problem in dictionary form.

    Methods:
    - policy_improvement(x3): returns policy improvement function value for given x3.
    - policy_valuation(x1): returns policy valuation function value for given x1.
    - classical_combinatorial_solution(max_iter=5): solves the combinatorial optimization 
        problem using a classical combinatorial algorithm.

    """
    def __init__(
                self,
                x1_init = 0.5,
                x2_init = -0.5,
                x3_init = 0.5
                ):
        """
        Initializes the `ClassicalCombinatorialSolution` object with the specified initial values 
        for x1, x2, and x3, and loads the QUBO dictionary from a file.

        Parameters:
        - x1_init (float), default=0.5: the initial value for x1.
        - x2_init (float), default=-0.5: the initial value for x2.
        - x3_init (float), default=0.5: the initial value for x3.

        """

        # Set initial values.
        self.x1_init = x1_init
        self.x2_init = x2_init
        self.x3_init = x3_init

        # Load QUBO.
        with open("./inputs/classical_combinatorial_qubo.pickle", "rb") as f:
            self.qubo = pickle.load(f)

    def __str__(self):
        """
        Returns a string representation of the ClassicalCombinatorialSolution object.

        Returns:
        - str: A string containing the current values of x1, x2, and x3.
        """
        return f"ClassicalCombinatorialSolution(x1_init={self.x1_init}, x2_init={self.x2_init}, x3_init={self.x3_init})"

    def policy_improvement(self, x3):
        """
        Performs the policy improvement step, updating x1.

        Args:
        - x3 (float): the current value of x3.

        Returns:
        - float: the value of x1 for a given value of x3.

        """

        return ALPHA*BETA*x3/(1+ALPHA*BETA*x3)

    def policy_valuation(self, x1):
        """
        Performs the policy valuation step, updating x2 and x3.

        Args:
        - x1 (float): the current value of x1.

        Returns:
        - Tuple of two floats: the values of x2 and x3 for a given value of x1.

        """

        # Define dictionary.
        updated_qubo = {}

        # Update qubo.
        for key, value in self.qubo.items():
            updated_qubo[key] = eval(str(value).replace('x1', str(x1)))   

        # Instantiate exact solver.
        sampler = dimod.ExactSolver()

        # Convert qubo dictionary.
        qubo_dict = dimod.BQM.from_qubo(updated_qubo)
        
        # Sample solution.
        sampleset = sampler.sample(qubo_dict)

        # Recover lowest loss sample.
        sample = sampleset.first.sample

        # Recover terminal states of binary variables.
        x2 = sum(S2 * sample[f'x2{i}'] * U2[i] for i in range(N2))
        x3 = sum(S3 * sample[f'x3{i}'] * U3[i] for i in range(N3))

        return x2, x3


    def classical_combinatorial_solution(self, max_iter = 5):
        """
        Solves the combinatorial optimization problem using a classical algorithm.

        Args:
        - max_iter (int), default=5: the maximum number of iterations to perform.

        Returns:
        - pandas.DataFrame: a DataFrame containing the error values and execution time for each iteration.

        """
        # Set initial values.
        x1_iterations = [self.x1_init]
        x2_iterations = [self.x2_init]
        x3_iterations = [self.x3_init]

        # Set initial time elapsed.
        time_iterations = [0]

        # Iteration.
        iteration = 1

        # Iterate over policy improvement and valuation steps.
        while iteration <= max_iter:
            # Start timer.
            start_time = time.time()
            
            # Perform policy improvement and valuation steps.
            x1 = self.policy_improvement(x3_iterations[iteration-1])
            x2, x3 = self.policy_valuation(x1_iterations[iteration-1])

            # Append results.
            x1_iterations.append(x1)
            x2_iterations.append(x2)
            x3_iterations.append(x3)

            # Record time elapsed.
            time_iterations.append(time.time() - start_time + time_iterations[iteration-1])

            # Increment iteration.
            iteration += 1

        # Compute errors.
        x1_error = [100*abs((x - X1_TRUE) / X1_TRUE) for x in x1_iterations]
        x2_error = [100*abs((x - X2_TRUE) / X2_TRUE) for x in x2_iterations]
        x3_error = [100*abs((x - X3_TRUE) / X3_TRUE) for x in x3_iterations]

        # Create results DataFrame.
        results = pd.DataFrame(np.vstack([x1_error, x2_error, x3_error, time_iterations]).T,\
                            columns = ['x1_error', 'x2_error', 'x3_error', 'total_time'])

        return results

if __name__ == '__main__':
    # Set number of exectutions.
    NUM_EXECUTIONS = 10

    # Define empty list to hold results.
    results_list = []

    # Create instance of class.
    solver = ClassicalCombinatorialSolution()

    # Compute exact solution.
    for i in range(NUM_EXECUTIONS):
        results_list.append(solver.classical_combinatorial_solution(max_iter = 3))

    # Concatenate dataframes.
    results_concat = pd.concat(results_list)

    # Compute mean results.
    mean_results = results_concat.groupby(pd.Grouper(level=0)).mean().round(2)

    # Clean data.
    mean_results['total_time'] = mean_results['total_time'].apply(lambda x: f"{x}E+06")



"""
This module implements a classical solution for a combinatorial optimization 
problem. It contains a class `ClassicalSolution` that provides methods for 
computing the classical solution using an iterative policy improvement and 
policy valuation process. The results of the computation are returned as 
a pandas DataFrame.

"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import brute

from constants.classical import (ALPHA, B3_FLAT, B4, BETA, LOG_Y_GRID_FLAT, M1,
                                 M2, M3, NUM_NODES, X1_TRUE, X2_TRUE, X3_TRUE)


class ClassicalSolution:
    """"
    Class that contains methods for computing the classical solution.

    Parameters:
    - x1_initial (float), optional: initial value of x1.
    - x2_initial (float), optional: initial value of x2.
    - x3_initial (float), optional: initial value of x3.

    Methods:
    - policy_improvement_function(self, x, x3): returns the policy improvement function value.
    - policy_valuation_function(self, x, x1): returns the policy valuation function value.
    - classical_solution(self, max_iter = 5): computes the classical solution.

    """

    def __init__(
                self,
                x1_init = 0.5,
                x2_init = -0.5,
                x3_init = 0.5
                ):
        """
        Initializes the class with initial values of x1, x2, and x3.
        
        Args:
        - x1_init (float): initial value of x1.
        - x2_init (float): initial value of x2.
        - x3_init (float): initial value of x3.

        """

        # Set initial values.
        self.x1_init = x1_init
        self.x2_init = x2_init
        self.x3_init = x3_init

    def __str__(self):
        """
        Returns a formatted string representation of the ClassicalSolution object.

        Returns:
        - str: a string representation of the ClassicalSolution object.
        
        """
        
        return (f"ClassicalSolution(x1_init={self.x1_init:.2f}, "
            f"x2_init={self.x2_init:.2f}, "
            f"x3_init={self.x3_init:.2f})")


    def policy_improvement_function(self, x, x3):
        """
        Returns the policy improvement function value.
        
        Args:
        - x (float): variable value for x1 (x[0]).
        - x3 (float): updated value of x3.
        
        Returns:
        - policy_improvement_loss (float): value of the policy improvement loss function.

        """

        policy_improvement_loss = -np.log(1-x[0])-B4*x3*np.log(x[0])

        return policy_improvement_loss

    def policy_valuation_function(self, x, x1):
        """
        Returns the policy valuation function value.
        
        Args:
        - x (float): variable values for x2 (x[0]) and x3 (x[1])
        - x1 (float): updated value of x1.
        
        Returns:
        - policy_valuation_loss (float): value of the policy valuation loss function.

        """

        # Compute policy valuation loss.
        policy_valuation_loss = np.sum((LOG_Y_GRID_FLAT + np.log(1 - x1) - (1 - BETA) * x[0] + \
                                    B3_FLAT * x[1] + ALPHA * BETA * np.log(x1) * x[1]) ** 2)

        return policy_valuation_loss

    def classical_solution(self, max_iter = 5):
        """
        Computes the classical solution.
        
        Args:
        - max_iter (int): maximum number of iterations.
        
        Returns:
        - results (DataFrame): DataFrame containing the computed results.

        """

        # Set values for initial iteration.
        x1_iterations = [self.x1_init]
        x2_iterations = [self.x2_init]
        x3_iterations = [self.x3_init]

        # Set time elapsed.
        time_iterations = [0]

        # Set iteration.
        iteration = 1

        # Initialize values.
        x1 = self.x1_init
        x2 = self.x2_init
        x3 = self.x3_init
 
        # Iterate over policy improvement and valuation updates.
        while iteration < max_iter:
           # Start timer.
            start_time = time.time()
            
            # Policy improvement step.
            x1 = brute(self.policy_improvement_function, [(1.0-M1, M1)],
                       args=(x3,), Ns = NUM_NODES, finish = None)
            
            # Policy valuation step.
            x2, x3 = brute(self.policy_valuation_function, [(M2, 0.0), (0.0, M3)],
                           args=(x1,), Ns = NUM_NODES, finish = None)

            # Record time.
            time_iterations.append(time.time() - start_time + time_iterations[iteration-1])

            # Append results to lists.
            x1_iterations.append(x1)
            x2_iterations.append(x2)
            x3_iterations.append(x3)

            # Increment iteration.
            iteration += 1

        # Compute errors.
        x1_error = [100*abs((x - X1_TRUE) / X1_TRUE) for x in x1_iterations]
        x2_error = [100*abs((x - X2_TRUE) / X2_TRUE) for x in x2_iterations]
        x3_error = [100*abs((x - X3_TRUE) / X3_TRUE) for x in x3_iterations]

        # Create DataFrame with results.
        results = pd.DataFrame(np.vstack([x1_error, x2_error, x3_error, time_iterations]).T,\
                                  columns = ['x1_error', 'x2_error', 'x3_error', 'total_time'])

        return results


if __name__ == '__main__':
    # Set number of executions.
    NUM_EXECUTIONS = 10

    # Create instance of class.
    solver = ClassicalSolution()

    # Define empty list for results.
    results_list = []

    # Compute results for each execution.
    for i in range(NUM_EXECUTIONS):
        results_list.append(solver.classical_solution(max_iter = 3))

    # Concatenate dataframes.
    results_concat = pd.concat(results_list)

    # Compute mean results.
    mean_results = results_concat.groupby(pd.Grouper(level=0)).mean().round(2)

    # Convert time to microseconds.
    mean_results['total_time'] = mean_results['total_time'].apply(lambda x: f"{x}E+06")


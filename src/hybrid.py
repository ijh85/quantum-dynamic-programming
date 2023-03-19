"""
This module implements a quantum-classical hybrid solution for the RBC model.
It provides the HybridSolution class, which contains methods needed to solve
the problem.

"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from constants.classical import ALPHA, BETA, X1_TRUE, X2_TRUE, X3_TRUE
from constants.hybrid import N2, N3, S2, S3, U2, U3
from utilities.utils import Binary


class HybridSolution:
    """
    A class that implements a quantum-classical hybrid solution.

    Parameters:
    - num_anneals (int): number of anneals to perform.
    - x1_init (float): initial value for x1.
    - x2_init (float): initial value for x2.
    - x3_init (float): initial value for x3.

    Methods:
    - policy_improvement(x3): performs policy improvement step for a given value of x3.
    - policy_valuation(x1): performs policy valuation step for a given value of x1.
    - convert_results(anneals): converts binary annealing results into floats.
    - hybrid_solution(): computes the hybrid solution.
    - compute_mean(anneals, variable_name): computes the mean value of a variable for the lowest energy anneals.
    - compute_error(true_value, solution_value): computes the error for selected variable.

    Attributes:
    - x1_init (float): initial value for x1.
    - x2_init (float): initial value for x2.
    - x3_init (float): initial value for x3.
    - max_iterations (int): maximum number of iterations.
    - num_anneals (int): number of anneals to perform per execution.
    - qubo (dict): the QUBO matrix in dictionary form.
    - sampler (FixedEmbeddingComposite): the sampler used for quantum annealing.

    """

    def __init__(self,
                 num_anneals,
                 x1_init = 0.5,
                 x2_init = -0.5,
                 x3_init = 0.5
                 ):
        """
        Initializes a HybridSolution object with the given parameters.

        Parameters:
        - num_anneals (int): number of anneals to perform.
        - x1_init (float): initial value for x1.
        - x2_init (float): initial value for x2.
        - x3_init (float): initial value for x3.

        """

        # Set initial values.
        self.x1_init = x1_init
        self.x2_init = x2_init
        self.x3_init = x3_init

        # Retain anneals below this energy percentile.
        self.anneal_percentile = 0.10

        # Set maximum number of iterations.
        self.max_iterations = 2

        # Load embedding.
        with open("./inputs/hybrid_embedding.pickle", "rb") as openfile:
            pegasus_embedding = pickle.load(openfile)

        # Load QUBO.
        with open("./inputs/hybrid_qubo.pickle", "rb") as f:
            self.qubo = pickle.load(f)

        # Instantiate sampler.
        qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})

        # Instantiate composite embedding.
        self.sampler = FixedEmbeddingComposite(qpu_advantage, pegasus_embedding)
        
        # Set number of anneals per execution.
        self.num_anneals = num_anneals

    def __str__(self):
        """
        Returns a formatted string representation of the HybridSolution object.

        Returns:
        - str: a string representation of the HybridSolution object.

        """
        
        return (f"HybridSolution(num_anneals={self.num_anneals}, "
            f"x1_init={self.x1_init:.2f}, "
            f"x2_init={self.x2_init:.2f}, "
            f"x3_init={self.x3_init:.2f})")

    def policy_improvement(self, x3):
        """
        Performs policy improvement step for a given value of x3.

        Parameters:
        - x3 (float): the updated value of x3.

        Returns:
        - float: The value of x1 for a given value of x3.

        """

        return ALPHA*BETA*x3/(1+ALPHA*BETA*x3)

    def policy_valuation(self, x1):
        """
        Returns a QUBO in dictionary form for a given value of x1.

        Parameters:
        - x1 (float): the value of x1 used in the policy valuation step.

        Returns:
        - dict: the updated QUBO used to compute x2 and x3.
        
        """

        # Define empty dictionary.
        updated_qubo = {}

        # Update QUBO.
        for key, value in self.qubo.items():
            updated_qubo[key] = eval(str(value).replace('x1', str(x1)))   

        return updated_qubo

    def convert_results(self, anneals):
        """
        Converts the binary annealing results to floats.

        Parameters:
        - anneals (pandas.DataFrame): the annealing results.

        Returns:
        - pandas.DataFrame: the annealing results converted to floats.
        
        """

        # Map binary results for x2 and x3 to float values.
        anneals['x2'] = sum(S2 * anneals[f'x2{i}'] * U2[i] for i in range(N2))
        anneals['x3'] = sum(S3 * anneals[f'x3{i}'] * U3[i] for i in range(N3))

        return anneals

    def hybrid_solution(self):
        """
        Computes the hybrid solution.

        Returns:
        - numpy.ndarray: the errors and execution times.

        """

        # Define lists to hold variables.
        x1_iterations = [self.x1_init]
        x2_iterations = [self.x2_init]
        x3_iterations = [self.x3_init]

        # Define lists to hold errors.
        x1_error = [self.compute_error(X1_TRUE, x1_iterations[0])]
        x2_error = [self.compute_error(X2_TRUE, x2_iterations[0])]
        x3_error = [self.compute_error(X3_TRUE, x3_iterations[0])]

        # Define lists for QPU times.
        qpu_access_time = [0]
        qpu_programming_time = [0]

        # Set initial time elapsed.
        total_time = [0]

        # Set iteration number.
        iteration = 1

        # Iterate over policy and valuation steps.
        while iteration <= self.max_iterations:
            # Initialize timer.
            start_time = time.time()

            # Policy improvement step: update x1.
            x1_update = self.policy_improvement(x3_iterations[iteration-1])

            # Policy valuation step: update x2 and x3.
            qubo_update = self.policy_valuation(x1_iterations[iteration-1])
            results_update = self.sampler.sample_qubo(qubo_update, num_reads = self.num_anneals)
            anneals_update = self.convert_results(results_update.to_pandas_dataframe())
            x2_update = self.compute_mean(anneals_update, 'x2')
            x3_update = self.compute_mean(anneals_update, 'x3')

            # Update time elapsed.
            total_time.append(time.time() - start_time + total_time[iteration-1])

            # Add updated values to lists.
            x1_iterations.append(x1_update)
            x2_iterations.append(x2_update)
            x3_iterations.append(x3_update)

            # Compute errors.
            x1_error.append(self.compute_error(X1_TRUE, x1_iterations[iteration]))
            x2_error.append(self.compute_error(X2_TRUE, x2_iterations[iteration]))
            x3_error.append(self.compute_error(X3_TRUE, x3_iterations[iteration]))

            # Update QPU access time.
            qpu_access_update = qpu_access_time[iteration-1] + results_update.info['timing']['qpu_access_time']/10**4
            qpu_access_time.append(qpu_access_update)

            # Update QPU programming time.
            qpu_programming_update = qpu_programming_time[iteration-1] + results_update.info['timing']['qpu_programming_time']/10**4
            qpu_programming_time.append(qpu_programming_update)

            # Update iteration number.
            iteration += 1

        # Collect results in numpy array.
        results = np.array([x1_error[-1], x2_error[-1], x3_error[-1],\
                            qpu_access_time[-1], qpu_programming_time[-1],\
                            total_time[-1]], float)

        return results

    def compute_mean(self, anneals, variable_name):
        """
        Computes the mean value of a variable for the lowest energy anneals.

        Parameters:
        - anneals (pandas.DataFrame): the annealing results.

        Returns:
        - float: the mean value of the selected variable.
        
        """

        # Compute mean of variable.
        variable_mean = anneals[anneals['energy'] < anneals['energy'].quantile(self.anneal_percentile)][variable_name].mean()

        return variable_mean
    
    def compute_error(self, true_value, solution_value):
        """
        Computes the error between the true and solution values.

        Parameters:
        - true_value (float): the true value.
        - solution_value (float): the solution value.

        Returns:
        - float: the computed error value.
        
        """
        
        # Compute error.
        error = 100*abs((solution_value - true_value) / true_value)

        return error

if __name__ == '__main__':
    # Number of executions.
    NUM_EXECUTIONS = 50

    # Define empty lists to hold results.
    result_list = []

    # Instantiate hybrid solver.
    solver = HybridSolution(num_anneals = 100)

    # Run solver.
    for i in range(NUM_EXECUTIONS):
        print(i)
        result_list.append(solver.hybrid_solution())
        time.sleep(1)

    # Convert list of lists to numpy array.
    result_list = np.vstack(result_list)

    # Compute summary statistics.
    mean_results = np.array([result_list.mean(axis=0),
          np.percentile(result_list, 25, axis = 0),
          np.percentile(result_list, 75, axis = 0),
          result_list.std(axis=0)]).round(2)

    # Convert to pandas dataframe.
    mean_results = pd.DataFrame(mean_results,\
                      columns = ['x1_error', 'x2_error',\
                         'x3_error', 'QPU Total',\
                        'QPU Prog.', 'Total Time'],\
                    index = ['mean','25%','75%','std'])

    # Convert times to microseconds.
    mean_results['QPU Total'] = mean_results['QPU Total'].apply(lambda x: f"{x}E+04")
    mean_results['QPU Prog.'] = mean_results['QPU Prog.'].apply(lambda x: f"{x}E+04")
    mean_results['Total Time'] = mean_results['Total Time'].apply(lambda x: f"{x}E+06")

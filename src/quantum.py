"""
The module contains the QuantumSolution class, which implements two quantum algorithms: 
multi_anneal and oneshot, both of which solve the RBC model.

The QuantumSolution class uses a quantum annealer to solve the dynamic programming problem by 
formulating it as a QUBO (quadratic unconstrained binary optimization) problem, which can be 
solved using quantum annealing. 

"""

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from dwave.cloud.exceptions import *
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from constants.classical import (B2, B3, B4, LOG_Y_GRID, X1_TRUE, X2_TRUE,
                                 X3_TRUE)
from constants.quantum import (A0, A0_, N1, N2, N3, P0, S1, S2, S3, U1, U2, 
                               U3, DELAY)
from utilities.utils import Binary


class QuantumSolution:
    """Class that implements two quantum algorithms: multi_anneal and oneshot.

    Args:
    - num_anneals (int): number of anneals to perform.
    - algorithm (str): algorithm to use for quantum annealing (oneshot or multi_anneal).

    Attributes:
    - algorithm (str): algorithm to use for quantum annealing.
    - num_anneals (int): number of anneals to perform per execution.
    - pegasus_embedding (dict): a dictionary containing the graph embedding.
    - qubo (dict): a dictionary containing the QUBO.
    - initialization (dict): a dictionary containing the initial values for the annealing process.
    - schedule (dict): a dictionary containing the annealing schedule.
    - offsets (dict): a dictionary containing the offsets for the annealing process.
    - qpu_advantage (DWaveSampler): a sampler object for the quantum annealing process.
    - sampler (FixedEmbeddingComposite): a fixed embedding composite object for the quantum annealing process.
    - reinitialization (bool): a flag indicating whether reinitialization is required for the annealing process.
    - annealing_threshold (int): number of lowest energy anneals to use when computing parameter values.

    Methods:
    - load_input_files(self): load input files based on algorithm.
    - convert_results(self, anneals): post-process raw annealing results.
    - compute_medians(self, anneals): get medians of x1 and x3.
    - compute_log_x1(self, x1): compute log median of x1.
    - correct_x2(self, anneals, log_x1_median, log_x1_median_, x3_median, x2s): get corrected x2.
    - corrected_x3(self, anneals, log_x1_median, log_x1_median_, x2_median, x3s): get corrected x3.
    - corrected_x1(self, anneals, log_x1_, log_x1, x3_median): get corrected x1.
    - mean_of_anneals(self, anneals): returns the mean of the lowest energy anneals.
    - compute_errors(self, anneals): compute errors.
    - min_energy_errors(self, anneals): returns errors for anneal with mininum energy.

    """
    def __init__(self,
                 num_anneals,
                 algorithm):
        """
        Initialize a QuantumSolution object.

        Args:
        - num_anneals (int): The number of anneals to be performed.
        - algorithm (str): The name of the algorithm: either 'multi_anneal' or 'oneshot'.

        Returns:
        - None. Initializes instance variables of the QuantumSolution object.

        """

        # Set algorithm type.
        self.algorithm = algorithm

        # Set number of anneals per execution.
        self.num_anneals = num_anneals

        # Load input files.
        self.load_input_files()

        # Set QPU type.
        self.qpu_advantage = DWaveSampler(solver={"qpu": True, "name": "Advantage_system6.1"})

        # Set embedding.
        self.sampler = FixedEmbeddingComposite(self.qpu_advantage, self.pegasus_embedding)

        # Set reinitialization flag conditional on algorithm.
        if self.algorithm == "multi_anneal":
            self.reinitialization = False
        elif self.algorithm == "oneshot":
            self.reinitialization = True

        # Set number of lowest energy anneals to use.
        self.annealing_threshold = 5

    def __str__(self):
        """Return a string representation of the QuantumSolution object."""
        output = f"QuantumSolution object with {self.num_anneals} anneals using {self.algorithm} algorithm\n"
        output += f"QPU type: {self.qpu_advantage}\n"
        output += f"Reinitialization flag: {self.reinitialization}\n"
        output += f"Annealing threshold: {self.annealing_threshold}\n"
        output += f"QUBO: {self.qubo}\n"
        output += f"Initial state: {self.initialization}\n"
        output += f"Annealing schedule: {self.schedule}\n"
        output += f"Offsets: {self.offsets}\n"
        return output

    def load_input_files(self):
        """
        Load input files based on the selected algorithm. This method loads the following files:
        - embedding_file: JSON file containing the embedding for the quantum annealer.
        - qubo_file: Pickle file containing the QUBO dictionary.
        - init_file: JSON file containing the initial state of the annealer.
        - sched_file: JSON file containing the annealing schedule for the quantum annealer.
        - off_file: Pickle file containing the offsets for the annealing schedule.

        Args:
        - self (QuantumSolution): the QuantumSolution object.

        Returns:
        - None: this method sets instance variables of the QuantumSolution object.

        """

        if self.algorithm == "multi_anneal":
            embedding_file = "./inputs/multi_anneal_quantum_embedding.json"
            qubo_file = "./inputs/multi_anneal_quantum_qubo.pickle"
            init_file = "./inputs/multi_anneal_quantum_initialization.json"
            sched_file = "./inputs/multi_anneal_schedule.json"
            off_file = "./inputs/multi_anneal_offsets.pickle"
        elif self.algorithm == "oneshot":
            embedding_file = "./inputs/oneshot_quantum_embedding.json"
            qubo_file = "./inputs/oneshot_quantum_qubo.pickle"
            init_file = "./inputs/oneshot_quantum_initialization.json"
            sched_file = "./inputs/oneshot_schedule.json"
            off_file = "./inputs/oneshot_offsets.pickle"
            
        # Load embedding.    
        with open(embedding_file, "r", encoding = "utf-8") as f:
            self.pegasus_embedding = json.load(f)
            
        # Load QUBO.
        with open(qubo_file, "rb") as f:
            self.qubo = pickle.load(f)

        # Load initialization.    
        with open(init_file, "r", encoding = "utf-8") as f:
            self.initialization = json.load(f)
            
        # Load annealing schedule.    
        with open(sched_file, "r", encoding = "utf-8") as f:
            self.schedule = json.load(f)

        # Load offsets.
        with open(off_file, "rb") as f:
            self.offsets = pickle.load(f)

        time.sleep(DELAY)

    def convert_results(self, anneals):
        """
        Convert raw results from the annealer to post-processed values.

        Args:
        - self (QuantumSolution): the QuantumSolution object.
        - anneals (DataFrame): a pandas DataFrame containing the results from the quantum annealer.

        Returns:
        - anneals (DataFrame): a pandas DataFrame containing the converted results and computed errors.

        """

        # Define empty lists for energy levels.
        corrected_value = []
        corrected_policy = []

        # Define empty lists for variables.
        x1_corrected = []
        x2_corrected = []
        x3_corrected = []

        # Compute intermediate values.
        x1_input = S1 * sum(anneals[f'x1{j}'] * U1[j] for j in range(N1))

        # Compute x1.
        x1 = P0[0] + P0[1]*x1_input

        # Compute x2.
        x2 = S2 * sum(anneals[f'x2{j}'] * U2[j] for j in range(N2))

        # Compute x3.
        x3 = S3 * sum(anneals[f'x3{j}'] * U3[j] for j in range(N3))

        # Update dataframe with calculated values.
        anneals['x1'], anneals['x2'], anneals['x3'] = x1, x2, x3

        # Compute ln(x1) and ln(1-x1).
        log_x1, log_x1_ = np.log(x1), np.log(1-x1)

        # Correct energy levels.
        for j in range(self.num_anneals):
            corrected_value.append(((LOG_Y_GRID + log_x1_[j] - B2 * x2[j] + B3 * x3[j] + \
                                     B4 * log_x1[j] * x3[j]) ** 2).mean())
            corrected_policy.append(-log_x1_[j] - B4 * log_x1[j] * x3[j])

        # Add corrected energy levels to anneals dataframe.
        anneals['corrected_value'] = corrected_value
        anneals['corrected_policy'] = corrected_policy

        # Perform post-processing for multi_anneal algorithm.
        if self.algorithm == "multi_anneal":

            # Get mean of variables for lowest energy anneals.
            x1_mean, x2_mean, x3_mean = self.mean_of_anneals(anneals)

            # Compute log of mean of x1.
            log_x1_mean, log_x1_mean_ = np.log(x1_mean), np.log(1-x1_mean)
            
            # Correct energy levels.
            for j in range(self.num_anneals):
                x1_corrected.append(-log_x1_mean_ - B4 * log_x1_mean * x3_mean)
                x2_corrected.append(((LOG_Y_GRID + log_x1_mean_ - B2 * x2[j] + B3 * x3_mean + \
                                      B4 * log_x1_mean * x3_mean) ** 2).mean())
                x3_corrected.append(((LOG_Y_GRID + log_x1_mean_ - B2 * x2_mean + B3 * x3[j] + \
                                      B4 * log_x1_mean * x3[j]) ** 2).mean())

            # Add corrected variables to anneals dataframe.
            anneals['x1_corrected'] = x1_corrected
            anneals['x2_corrected'] = x2_corrected
            anneals['x3_corrected'] = x3_corrected

            # Add variable errors to results dataframe.
            anneals = self.compute_errors(anneals)

        # Perform post-processing for oneshot algorithm.
        if self.algorithm == "oneshot":

            # Compute median of anneals.
            x1_median, x3_median = self.compute_medians(anneals)

            # Compute logs.
            log_x1_median, log_x1_median_ = np.log(x1_median), np.log(1-x1_median)

            # Set iteration.
            iteration = 0

            # Correct variables.
            while iteration <= 1:
                # Update x1, x2, and x3.
                x2_median, anneals = self.correct_x2(anneals, log_x1_median, log_x1_median_, x3_median, x2)
                x3_median, anneals = self.correct_x3(anneals, log_x1_median, log_x1_median_, x2_median, x3)
                x1_median, log_x1_median, log_x1_median_, anneals = self.correct_x1(anneals, log_x1, log_x1_, x3_median)
                
                # Update iteration.
                iteration += 1

            # Add variable errors to results dataframe.
            anneals = self.compute_errors(anneals)

        return anneals

    def compute_medians(self, anneals):
        """
        Returns the median value of x1 and x3 for the best anneals.

        Args:
        - self (QuantumSolution): the QuantumSolution object.
        - anneals (DataFrame): a pandas DataFrame containing the results of the quantum annealer.

        Returns:
        - x1_median (float): the median value of x1 for the best anneals.
        - x3_median (float): the median value of x3 for the best anneals.

        """

        # Get median of x1 and x3 for low energy anneals.
        x1_median = anneals.nsmallest(self.annealing_threshold, 'corrected_value')['x1'].median()
        x3_median = anneals.nsmallest(self.annealing_threshold, 'corrected_value')['x3'].median()
        
        return x1_median, x3_median

    def compute_log_x1(self, x1):
        """Get the natural logarithm of the median of x1 and 1-x1 values.

        Args:
        - x1 (float): the x1 value.

        Returns:
        - tuple: a tuple of two numpy arrays containing the natural logarithms of the
            x1 and 1-x1 values.

        """
        
        # Get natural logarithm of x1.
        log_x1 = A0[0] + A0[1] * x1 + A0[2] * x1**2
        
        # Get natural logarithm of 1-x1.
        log_x1_ = A0_[0] + A0_[1] * x1
        
        return log_x1, log_x1_

    def correct_x2(self, anneals, log_x1_median, log_x1_median_, x3_median, x2):
        """Compute the corrected value for x2.

        Args:
        - anneals (pandas.DataFrame): the anneal data with the 'corrected_value', 'corrected_policy',
            'corrected_combined', 'value_rank', 'policy_rank', 'combined_rank', and 'max_rank' columns.
        - log_x1_median (numpy.ndarray): the median natural logarithm of the x1 values.
        - log_x1_median_ (numpy.ndarray): the median natural logarithm of the 1-x1 values.
        - x3_median (float): the median value of x3.
        - x2 (numpy.ndarray): the array of x2 values.

        Returns:
        - float: the median value of x2.
        - pandas.DataFrame: the updated anneals dataframe with the 'x2_corrected' column.

        """

        # Define empty list to hold corrected values.
        x2_corrected = []
        
        # Compute corrected value for x2.
        for j in range(self.num_anneals):
            x2_corrected.append(((LOG_Y_GRID + log_x1_median_ - B2 * x2[j] +
                              B3 * x3_median + B4 * log_x1_median * x3_median)**2).mean())
        
        # Add corrected values to anneals dataframe.
        anneals['x2_corrected'] = x2_corrected
        
        # Compute median of corrected values.
        x2_median = anneals.nsmallest(self.annealing_threshold, 'x2_corrected')['x2'].median()
        
        return x2_median, anneals

    def correct_x3(self, anneals, log_x1_median, log_x1_median_, x2_median, x3):
        """Compute the corrected value for x3.

        Args:
        - anneals (pandas.DataFrame): The anneal data with the 'corrected_value', 'corrected_policy',
            'corrected_combined', 'value_rank', 'policy_rank', 'combined_rank', and 'max_rank' columns.
        - log_x1_median (numpy.ndarray): The median natural logarithm of the x1 values.
        - log_x1_median_ (numpy.ndarray): The median natural logarithm of the 1-x1 values.
        - x2_median (float): The median value of x2.
        - x3 (numpy.ndarray): The array of x3 values.

        Returns:
        - float: The median value of x3.
        - pandas.DataFrame: The updated anneals data frame with the 'x3_corrected' column.

        """

        # Define empty list to hold corrected values.
        x3_corrected = []

        # Compute corrected value for x3.
        for j in range(self.num_anneals):
            x3_corrected.append(((LOG_Y_GRID + log_x1_median_ - B2 * x2_median +
                              B3 * x3[j] + B4 * log_x1_median * x3[j])**2).mean())
        

        # Add corrected values to anneals dataframe.
        anneals['x3_corrected'] = x3_corrected
        
        # Compute median of corrected values.
        x3_median = anneals.nsmallest(self.annealing_threshold, 'x3_corrected')['x3'].median()
        
        return x3_median, anneals

    def correct_x1(self, anneals, log_x1, log_x1_, x3_median):
        """Corrects x1 based on the median value of x3.

        Args:
        - anneals (pandas.DataFrame): dataframe containing annealing results.
        - log_x1 (numpy.ndarray): array of the natural logarithm of x1.
        - log_x1_ (numpy.ndarray): array of the natural logarithm of 1 - x1.
        - x3_median (float): median value of x3.

        Returns:
        - tuple: contains the corrected x1 value, natural logarithm of x1, 
            natural logarithm of 1-x1, and the updated annealing results DataFrame.

        """

        # Define empty list to hold values.
        x1_corrected = []
        
        # Compute corrected value for x1.
        for j in range(self.num_anneals):
            x1_corrected.append((-log_x1_[j] - B4 * log_x1[j] * x3_median).mean())
        
        # Add corrected values to anneals dataframe.
        anneals['x1_corrected'] = x1_corrected
        
        # Compute median of corrected values.
        x1_median = anneals.nsmallest(self.annealing_threshold, 'x1_corrected')['x1'].median()
        
        # Compute natural logarithm of x1 and 1-x1.
        log_x1_median = A0[0] + A0[1] * x1_median + A0[2] * x1_median**2
        
        # Get natural logarithm of 1-x1.
        log_x1_median_ = A0_[0] + A0_[1] * x1_median
        
        return x1_median, log_x1_median, log_x1_median_, anneals

    def mean_of_anneals(self, anneals):
        """Computes the mean values of the best anneals.

        Args:
        - anneals (pandas.DataFrame): dataframe containing annealing results.
        - number_anneals (int): number of top anneals to consider.

        Returns:
        - tuple: tuple containing the mean x1, mean x2, and mean x3 values.

        """
        
        # Compute mean variable value for lowest energy anneals.
        mean_x1 = anneals.nsmallest(self.annealing_threshold, 'corrected_value')['x1'].mean()
        mean_x2 = anneals.nsmallest(self.annealing_threshold, 'corrected_value')['x2'].mean()
        mean_x3 = anneals.nsmallest(self.annealing_threshold, 'corrected_value')['x3'].mean()
        
        return mean_x1, mean_x2, mean_x3

    def compute_errors(self, anneals):
        """Computes the errors between the annealing results and the true values.

        Args:
        - anneals (pandas.DataFrame): Dataframe containing annealing results.

        Returns:
        - pandas.DataFrame: Dataframe containing the updated annealing results with 
            error values for x1, x2, and x3.

        """
        
        # Compute errors for variables.
        anneals['x1_error'] = 100 * np.abs((anneals['x1'] - X1_TRUE) / X1_TRUE)
        anneals['x2_error'] = 100 * np.abs((anneals['x2'] - X2_TRUE) / X2_TRUE)
        anneals['x3_error'] = 100 * np.abs((anneals['x3'] - X3_TRUE) / X3_TRUE)
        
        return anneals

    def min_energy_errors(self, anneals):
        """Compute the error values of the minimum energy anneal.

        Args:
        - anneals (pandas.DataFrame): a DataFrame containing annealing results.

        Returns:
        - tuple: a tuple containing the error values for x1, x2, and x3.

        """

        error_min_x1 = anneals.nsmallest(self.annealing_threshold, 'x1_corrected')['x1'].mean() 
        error_min_x2 = anneals.nsmallest(self.annealing_threshold, 'x2_corrected')['x2'].mean()
        error_min_x3 = anneals.nsmallest(self.annealing_threshold, 'x3_corrected')['x3'].mean()
        
        return error_min_x1, error_min_x2, error_min_x3

if __name__ == '__main__':
    # Set number of executions.
    NUM_EXECUTIONS = 2

    # Define list to collect results.
    results_list = []
    error_list = []
    qpu_access_time = []
    qpu_programming_time = []
    anneals_list = []

    # Instantiate solver.
    solver = QuantumSolution(num_anneals = 200,
                             algorithm = 'oneshot')
    
    # Define reverse annealing parameters.
    reverse_anneal_params = dict(anneal_schedule=solver.schedule,\
                             initial_state=solver.initialization,\
                             reinitialize_state=solver.reinitialization)

    # Loop over number of executions.
    for i in range(NUM_EXECUTIONS):
        print(i)
        results_list.append(solver.sampler.sample_qubo(solver.qubo,\
                                                 num_reads=solver.num_anneals,\
                                                 anneal_offsets=solver.offsets,\
                                                 **reverse_anneal_params))
        time.sleep(1)

    # Post-process results.
    for i in range(NUM_EXECUTIONS):
        anneals_list.append(solver.convert_results(results_list[i].to_pandas_dataframe()))
        error_list.append(solver.min_energy_errors(anneals_list[i]))
        qpu_access_time.append(results_list[i].info['timing']['qpu_access_time']/10**4)
        qpu_programming_time.append(results_list[i].info['timing']['qpu_programming_time']/10**4)

    # Construct results dataframe.
    mean_results = pd.DataFrame(error_list, columns = ['x1_error', 'x2_error', 'x3_error'])
    mean_results['qpu_access_time'] = qpu_access_time
    mean_results['qpu_programming_time'] = qpu_programming_time

    # Clean data.
    mean_results = mean_results.describe(percentiles=[0.25, 0.75]).loc[['mean','25%','75%','std']].round(5)
    mean_results['qpu_access_time'] = mean_results['qpu_access_time'].apply(lambda x: f"{x}E+04")
    mean_results['qpu_programming_time'] = mean_results['qpu_programming_time'].apply(lambda x: f"{x}E+04")



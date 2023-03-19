"""
This file contains constants for the classical combinatorial solution.
"""

# Set number of binary variables.
N2 = 10
N3 = 10

# Define binary coefficients.
U2 = [2**j for j in range(N2)]
U3 = [2**j for j in range(N3)]

# Set maximum values.
M2 = -36.0
M3 = 3.0

# Define scale factors.
S2 = M2/(2**N2-1)
S3 = M3/(2**N3-1)
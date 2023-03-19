"""
This file contains constants for the quantum solutions.
"""

from constants.classical import X1_TRUE

# Set number of binary variables.
N1 = 7
N2 = 7
N3 = 7

# Scaling coefficients for x1.
P0 = [0.25, 0.15]

# Define polynomial parameters for log approx.
A0 = [-2.64699173, 6.26193702, -4.84735673]
A0_ = [0.08755102, -1.48523314]

# Define binary coefficients.
U1 = [2**j for j in range(N1)]
U2 = [2**j for j in range(N2)]
U3 = [2**j for j in range(N3)]

# Set maximum values.
M1 = 1.0
M2 = -36.0
M3 = 3.0

# Define scale factors.
S1 = M1/(2**N1-1)
S2 = M2/(2**N2-1)
S3 = M3/(2**N3-1)

# Loading delay.
DELAY = 1.0

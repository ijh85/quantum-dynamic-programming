"""
This file contains constants for the classical solution.
"""

import numpy as np

# Define constants.
ALPHA = 1.0/3.0
BETA = 0.95
B = ALPHA/(1-BETA*ALPHA)
A = (np.log(1/(1+BETA*B))+\
            (BETA*B*np.log((BETA*B)/(1+BETA*B))))/(1-BETA)

# Productivity values.
Z = np.array([0.9792, 0.9896, 1.0000, 1.0106, 1.0212])
NZ = len(Z)

# Transition matrix.
ZTM = np.array([[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
            [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
            [0.0000, 0.0082, 0.9837, 0.0082, 0.0000],
            [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
            [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]])

# Set steady state values.
KSS = (ALPHA*BETA)**(1/(1-ALPHA))
YSS = KSS**ALPHA
CSS = YSS-KSS

# Define true values.
X1_TRUE = 0.3143542643901046
X2_TRUE = -18.284056758341677
X3_TRUE = 1.4478286454871008

# Compute state vector.
NK = 100
K = np.linspace(0.5*KSS, 1.5*KSS, NK)

# Compute expected productivity.
EZ = np.dot(Z, ZTM.T)
EZ = np.tile(EZ, (NK, 1))
LOG_EZ = np.log(EZ)

# Precompute variable grids.
K_GRID = np.tile(K, (NZ, 1)).T
Z_GRID = np.tile(Z, (NK, 1))
Y_GRID = np.multiply(pow(K_GRID, ALPHA), Z_GRID)
LOG_Y_GRID = np.log(Y_GRID)

# Define coefficients.
B0 = LOG_Y_GRID
B2 = (1-BETA)
B3 = (BETA*LOG_EZ+(ALPHA*BETA-1)*LOG_Y_GRID)
B4 = ALPHA*BETA
B3_ = BETA*(LOG_EZ+ALPHA*LOG_Y_GRID)

# Flatten grids.
LOG_Y_GRID_FLAT = LOG_Y_GRID.flatten()
B3_FLAT = B3.flatten()

# Set maximum values.
M1 = 0.99
M2 = -36.0
M3 = 3.0

# Number of nodes.
NUM_NODES = 2**7

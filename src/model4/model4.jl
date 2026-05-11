## Setup

include("utilities.jl")

## Pipeline overview

# 1. Load in data
# 2. Determine hyperparameters -
#   bin size
#   sliding window size
#   sliding window step
#   PCA pctage cutoff
#   hidden state dimension
# 3. loop over condition
#   3a. determine train / test / val split
#   3b. loop over windows
#       3a1. Fit PCA to window on train set
#       3a2. Determine how many dims cutoff is at with test set
#       3a3. fit dynamical system on PCA'd data at that dimension

## Pipeline implementation

p = Pickle.npyload("gdrive/mc_pacman.pkl")

# hyperparameters

binsize = 20
windowsize = 5 # in multiples of binsize
windowstep = 1 # in multiples of binsize
PCA_pctage = 0.8
max_hidden = 12 # no larger than number of components -2 either!

# this model doesn't seem very bayesian.....

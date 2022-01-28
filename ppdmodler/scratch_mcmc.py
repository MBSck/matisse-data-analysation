#!/usr/bin/env python3

"""Test file for a multidimensional Gaussian fit with MCMC; The emcee
package"""

import numpy as np
import emcee
import matplotlib.pyplot as plt

# Logarithm of p -> log_prob
def log_prob(x, mu, cov):
    """The Logaritm of p

    Parameters
    ----------
    p: List[int | float]
        The density p. A list of walkers from which a singular one is extracted
    mu: int | float
        Parameter of Gaussian
    cov: int | float
        Parameter of Gaussian

    Returns
    -------
    """
    diff = x - mu
    return -0.5*np.dot(diff, np.linalg.solve(cov, diff))

# Sets the steps of the burn-in and the max. steps
steps_bi, steps = 100, 10000

# The number of walkers (must be even) and the number of dimensions/parameters
nwalkers, ndim = 32, 5

# hyperparameters in 5-dimension
np.random.seed(42)
means = np.random.rand(ndim)

cov = 0.5-np.random.rand(ndim**2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T-np.diag(cov.diagonal())
cov = np.dot(cov, cov)

# This vector defines the starting points of each walker for the amount of
# dimensions
p0 = np.random.rand(nwalkers, ndim)

# The EnsambleSampler gets the parameters. The args are the args put into the
# lob_prob function. Additional parameter a can be used for the stepsize. None is
# the default 
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov], a=6)

# Burn-in of the sampler. Explores the parameter space. The walkers get settled
# into the maximum of the density. Saves the walkers in the state variable
print("Burn-in")
state = sampler.run_mcmc(p0, steps_bi)

# Resets the chain to remove burn in samples
sampler.reset()

# Do production. Starts from the final position of burn-in chain (rstate0 is
# the state of the internal random number generator)
print("Production")
sampler.run_mcmc(p0, steps, rstate0=state)

# This gets the parameter values for every walker at each step in the chain.
# Array of shape (steps, nwalkers, ndim)
samples = sampler.get_chain(flat=True)
print(samples)

# Plot the samples to get estimate of the density that has been sampled, to
# test if sampling went well
plt.hist(samples[:, 0], 100, color='k', histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([]);
plt.show()

# Another test of whether or not the sampling went well is the mean acceptance
# fraction and the integrated autocorrelation time
# Acceptance fraction has an entry for each walker -> It is a vector with the
# dimensions of the steps
# As a rule of thumb if the acceptance_fraction is below 0.2, the a parameter
# needs to be decreased, and if it is above 0.5 then the a parameter needs to
# be increased
acceptance = np.mean(sampler.acceptance_fraction)

# The autocorrelation time is a vector with ndim dimensions
autocorr= np.mean(sampler.get_autocorr_time())
print(f"Mean acceptance fraction {acceptance} and the autcorrelation time {autocorr}")

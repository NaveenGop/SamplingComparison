import scipy.stats as stats
import math
import matplotlib.pyplot as plt

import numpy as np
from scipy import integrate

#######################################
###   PDFs of three distributions   ###
#######################################

def normal(mu,sigma):
    return lambda x: stats.norm.pdf(x,loc = mu, scale = sigma)

def exp(lam):
    return lambda x: lam*math.exp(-lam*x) if x >=0 else 0

def gauss_mix(p,mu1,sig1,mu2,sig2):
    """
    Gaussian mixture with probabilities of selection being p and 1-p for N(mu1,sig1) and N(mu2,sig2) respectively
    """

    return lambda x: p*stats.norm.pdf(x,loc = mu1, scale = sig1) + (1-p)*stats.norm.pdf(x, loc = mu2, scale = sig2)

def total_variation_distance(samples1, samples2, dt):
    """ Romberg integration using samples of the density functions.
    Ensure that exactly 2 ** k + 1 evenly spaced samples are created."""
    assert samples1.shape == samples2.shape, 'different sample lengths'
    result = 0.5 * integrate.romb(np.abs(samples1 - samples2), dt)
    while type(result) == np.ndarray:
        result = integrate.romb(result, dt)
    return result

#######################################
###   Plotting and Graping Utils    ###
#######################################

def plot_histogram_and_transitions(samples):
    plt.hist(samples)
    plt.show()
    plt.plot([i for i in range(len(samples))], samples)
    plt.show()

def plot_transitions(samples):
    plt.plot([i for i in range(len(samples))], samples)
    plt.show()

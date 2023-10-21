import numpy as np

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / (sigma+0.000000001), mu, sigma
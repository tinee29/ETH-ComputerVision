import numpy as np

def estimate(particles, particles_w):
    # particles: nx2
    # particles_w: nx1
    
    # Transpose particles and perform matrix multiplication
    # Result: 1x2
    mean_state = np.dot(particles.T, particles_w).T
    
    return mean_state


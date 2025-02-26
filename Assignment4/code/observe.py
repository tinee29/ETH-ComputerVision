import numpy as np
from chi2_cost import chi2_cost  # Assuming you have a chi2_cost.py file
from color_histogram import color_histogram
def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist_target, sigma_observe):
    num_particles = particles.shape[0]
    particles_w = np.zeros(num_particles)

    xMin = particles[:, 0] - bbox_width / 2
    xMax = particles[:, 0] + bbox_width / 2
    yMin = particles[:, 1] - bbox_height / 2
    yMax = particles[:, 1] + bbox_height / 2

    for i in range(num_particles):
        hist_i = color_histogram(xMin[i], yMin[i], xMax[i], yMax[i], frame, hist_bin)
        chi_dist = chi2_cost(hist_target, hist_i)

        particles_w[i] = 1 / (np.sqrt(2 * np.pi) * sigma_observe) * np.exp(-0.5 * chi_dist**2 / (sigma_observe**2))

    particles_w = particles_w / np.sum(particles_w)

    return particles_w

# Example Usage:
# particles_w = observe(particles, frame, bbox_height, bbox_width, hist_bin, hist_target, sigma_observe)

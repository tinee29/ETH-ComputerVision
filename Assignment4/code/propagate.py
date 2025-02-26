import numpy as np

def propagate(particles, frame_height, frame_width, params):
    num_particles = params["num_particles"]
    state_length = particles.shape[1]
    dt = 1

    sigma_p = params["sigma_position"]
    sigma_v = params["sigma_velocity"]

    particles = particles.T

    if params["model"] == 0:  # no-motion model
        A = np.array([[1, 0],
                      [0, 1]])
        noise = np.array([sigma_p, sigma_p])

    elif params["model"] == 1:  # constant velocity motion model
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        noise = np.array([sigma_p, sigma_p, sigma_v, sigma_v])

    for i in range(num_particles):
        w = noise * np.random.randn(1, state_length)
        particles[:, i] = np.dot(A, particles[:, i]) + w.flatten()

        center_particle_x = particles[0, i]
        center_particle_y = particles[1, i]

        if center_particle_x > frame_width:
            particles[0, i] = frame_width
        elif center_particle_x < 1:
            particles[0, i] = 1

        if center_particle_y > frame_height:
            particles[1, i] = frame_height
        elif center_particle_y < 1:
            particles[1, i] = 1

    particles = particles.T

    return particles

# This is the file for storing some data and constants
import numpy as np

G = 6.67430e-20 # km^3 / (kg^1.s^2)

Earth = {
    'name' : 'Earth',
    'mass' : 5.9721e24, # kg
    'mu' : 398600.4418, # km^3 / s^2
    'radius' : 6378.137, # km
    'J2' : 1.082635854e-3,
    'J3' : -2.5326613168e-6,
    'at_rot_vec' : np.array([0,0,7.292115146706979e-5]) # rad / s
}

CubeSat_3U = {
    'name' : 'CubeSat_3U',
    'mass' : 4, # kg
    'shape' : 'cube', 
    'sigma_N' : 1,
    'sigma_T' : 1,
    'T_wall' : 300, # k
    'max_slew_rate' : (3 * (np.pi/180)), # rad/s
    'lx' : 0.1, # m
    'ly' : 0.1, # m
    'lz' : 0.3, # m
    'covarience_matrix' : np.diag([0.1, 0.1, 0.1]), # km^2
}

CubeSat_12U = {
    'name' : 'CubeSat_12U',
    'mass' : 16, # kg
    'shape' : 'cube', 
    'sigma_N' : 1,
    'sigma_T' : 1,
    'T_wall' : 300, # k
    'max_slew_rate' : (3 * (np.pi/180)), # rad/s
    'lx' : 0.2, # m
    'ly' : 0.2, # m
    'lz' : 0.3, # m
    'covarience_matrix' : np.diag([0.1, 0.1, 0.1]), # km^2
}

CubeSat_16U = {
    'name' : 'CubeSat_16U',
    'mass' : 22, # kg
    'shape' : 'cube', 
    'sigma_N' : 1,
    'sigma_T' : 1,
    'T_wall' : 300, # k
    'max_slew_rate' : (3 * (np.pi/180)), # rad/s
    'lx' : 0.2, # m
    'ly' : 0.2, # m
    'lz' : 0.4, # m
    'covarience_matrix' : np.diag([0.1, 0.1, 0.1]), # km^2
}

Debri = {
    'name' : 'Debri',
    'mass' : 30, # kg
    'radius' : 1,
    'shape' : 'sphere_debri', 
    'C_d' : 2.2,
    'covarience_matrix' : np.diag([0.1, 0.1, 0.1]), # km^2
}
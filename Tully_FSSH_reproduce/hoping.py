import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splder
from Potential_and_der_tempsave_origin import SAC_potential, DAC_potential, ECR_potential
from Force_dij import Cal_dij_and_force
from RK4 import RK4, f

def hoping_and_adjust(dt, rho, velocity, dij , mass, state, eig):
    b12 = -2.0 * (rho[0][1].conjugate() * velocity * dij[0][1]).real
    b21 = -2.0 * (rho[1][0].conjugate() * velocity * dij[1][0]).real
    #print("b12: {0}, b21: {1}".format(b12, b21))
    #print("rho11: {0}, rho22: {1}".format(rho[0][0], rho[1][1]))
    #print("rho:",rho)
    g12 = dt * b21 / (rho[0][0].real)
    g21 = dt * b12 / (rho[1][1].real)
    #print("g12: {0}, g21: {1}".format(g12, g21))
    random_number = np.random.rand()
    if state == 0: # 这里，我们还要设定state=0是基态，state=1是激发态
        if random_number > g12:
            hop = False
        elif 0.5 * mass * velocity ** 2 < (eig[1] - eig[0]):
            hop = False
        else:
            hop = True
    else:
        if random_number > g21:
            hop = False
        else :
            hop = True
        
    if hop == True:
        if state == 0:
            state = 1
            new_K = 0.5 * mass * velocity ** 2 - (eig[1] - eig[0])
            #if new_K < 0:
                #print("K: {0}, new_K: {1}".format(0.5 * mass * velocity ** 2, new_K))
            velocity_orien = velocity / np.linalg.norm(velocity)
            velocity = np.sqrt(2 * new_K / mass)
            velocity = velocity * velocity_orien
        else:
            state = 0
            new_K = 0.5 * mass * velocity ** 2 + (eig[1] - eig[0])
            K = 0.5 * mass * velocity ** 2
            #if new_K < K:
                #print("K: {0}, new_K: {1}, state:{2}".format(0.5 * mass * velocity ** 2, new_K, state))
                #print(eig)
            velocity_orien = velocity / np.linalg.norm(velocity)
            velocity = np.sqrt(2 * new_K / mass)
            velocity = velocity * velocity_orien
            
    
    return state, velocity, hop
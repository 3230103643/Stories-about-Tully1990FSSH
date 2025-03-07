import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splder
from Potential_and_der_tempsave_origin import SAC_potential, DAC_potential, ECR_potential
from Force_dij import Cal_dij_and_force

def f(input, mass, force, dij, state, eig):
    # RK4前置准备
    # mass即核质量，force即核在目前电子态下的受力， dij即目前的非绝热耦合矩阵，state即当前电子态
    pos, velocity, rho11, rho12, rho22 = input
    now_force = force[state]
    out_rho11 = -2.0 * (rho12.conjugate() * velocity * dij[0][1]).real
    out_rho22 = 2.0 * (rho12 * velocity * dij[0][1].conjugate()).real
    out_rho12 = -(0+1j) * rho12 * (eig[0]-eig[1]) + rho11 * velocity * dij[0][1] - rho22 * velocity * dij[0][1]
    output = [velocity, now_force/mass, out_rho11, out_rho12, out_rho22]
    return output

def RK4(initial_con, dt, mass, force, dij, state, eig):
    pos, velocity, rho11, rho12, rho22 = initial_con

    k1 = f([pos, velocity, rho11, rho12, rho22], mass, force, dij, state, eig)
    k2 = f([pos + 0.5*dt*k1[0], velocity + 0.5*dt*k1[1], rho11 + 0.5*dt*k1[2], rho12 + 0.5*dt*k1[3], rho22 + 0.5*dt*k1[4]], mass, force, dij, state, eig)
    k3 = f([pos + 0.5*dt*k2[0], velocity + 0.5*dt*k2[1], rho11 + 0.5*dt*k2[2], rho12 + 0.5*dt*k2[3], rho22 + 0.5*dt*k2[4]], mass, force, dij, state, eig)
    k4 = f([pos + dt*k3[0], velocity + dt*k3[1], rho11 + dt*k3[2], rho12 + dt*k3[3], rho22 + dt*k3[4]], mass, force, dij, state, eig)

    pos_new = pos + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    velocity_new = velocity + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    rho11_new = rho11 + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    rho12_new = rho12 + (dt/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    rho22_new = rho22 + (dt/6)*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])

    return pos_new, velocity_new, rho11_new, rho12_new, rho22_new


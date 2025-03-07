import numpy as np
import matplotlib.pyplot as plt
from Potential_and_der_tempsave_origin import SAC_potential, DAC_potential, ECR_potential
from Force_dij import Cal_dij_and_force
from RK4 import RK4
from hoping import hoping_and_adjust
import time

# 参数初始化
testing_mode = int(input("Enter 1 for SAC, 2 for DAC, 3 for ECR: "))
left_border, right_border = -10, 10 # atomic units,下面也都是
mass, dt, num_traj = 2000, 1, 1000 # mass是质量，dt是时间步长，num_traj是系综轨道数

# 记录每个动量对应的TL_prob, TU_prob, RL_prob
#p0_values = [1.42, 4.12, 4.539, 4.550, 5.550, 5.687, 7.566, 7.811, 8.389, 9.38389, 9.810, 11.115, 12.940, 14.360, 15.924, 19.621, 22.607, 25.308, 27.726, 30.0]
p0_values = np.linspace(1.42, 35,100)
TL_probs = []
RL_probs = []
RU_probs = []
TU_probs = []
#phi11s = []
#phi12s = []
#phi21s = []
#phi22s = []
phi11pre = 100
#phi12pre = 100
phi21pre = 100
#phi22pre = 100

# 高斯分布的标准差
sigma = 1.5


for p0 in p0_values:
    start_time = time.time()
    TL, RU, RL, TU = 0, 0, 0, 0 
    p0_gaussian = np.random.normal(loc=p0, scale=sigma, size=num_traj)
    for idx, p1 in enumerate(p0_gaussian):
        v0 = p1 / mass
        pos_0 = left_border
    #v0 = p0 / mass # 初始速度
    #pos_0 = left_border # 初始位置
    #TL, RU, RL = 0, 0, 0 

    # 开搞
    #start_time = time.time()
    #for trajectory in range(num_traj):
        print(f"p1: {p1}, trajectory: {idx}")
        Max_step = 5000000 # 防止循环不收敛
        state = 0 # 最初电子位于基态, state=0是基态，state=1是激发态
        p, velocity, position = p1, v0, pos_0
        if testing_mode == 1:
            V, V_der = SAC_potential(position)
        elif testing_mode == 2:
            V, V_der = DAC_potential(position)
        else:
            V, V_der = ECR_potential(position)
        rho = np.array([[1+0j, 0+0j], [0+0j, 0+0j]]) # 密度矩阵初始化，最初电子位于基态
        eig, dij, force, vector = Cal_dij_and_force(V, V_der)
        phi11pre = 100
        #phi12pre = 100
        phi21pre = 100
        #phi22pre = 100

        while position <= right_border and position >= left_border and Max_step > 0:
            reverse_1, reverse_2 = 0, 0
            Max_step -= 1
            position, velocity, rho[0, 0], rho[0, 1], rho[1, 1] = RK4([position, velocity, rho[0, 0], rho[0, 1], rho[1, 1]], dt, mass, force, dij, state, eig) # 更新这一步长演化后的位置、速度、密度矩阵
            rho[1, 0] = rho[0, 1].conjugate()
            if testing_mode == 1: # 更新势能和势能导数
                V, V_der = SAC_potential(position)
            elif testing_mode == 2:
                V, V_der = DAC_potential(position)
            else:
                V, V_der = ECR_potential(position)
            eig, dij, force, vector = Cal_dij_and_force(V, V_der) # 更新下一步演化所受的力，以及dij、本征态势能
            phi1 = vector[:, 0]
            phi2 = vector[:, 1]
            if phi11pre != 100:
                if (phi11pre * phi1[0] < 0) and (abs(phi11pre - phi1[0]) / 2 > 0.05) : # 防止特征向量的正负号突然翻转
                    reverse_1 = 1
                    #print(f"phi1: {phi1}")
                    phi1[0] = -phi1[0]
                    phi1[1] = -phi1[1]
                    #print(f"phi1: {phi1}") 
            if phi21pre != 100:
                if (phi21pre * phi2[0] < 0) and (abs(phi21pre - phi2[0]) / 2 > 0.05) : # 防止特征向量的正负号突然翻转
                    reverse_2 = 1
                    #print(f"phi2: {phi2}") 
                    phi2[0] = -phi2[0]
                    phi2[1] = -phi2[1]
                    #print(f"phi2: {phi2}")
            if reverse_1 + reverse_2 == 1:
                dij[0, 1] = -dij[0, 1]
                dij[1, 0] = -dij[1, 0]
            state, velocity, hop = hoping_and_adjust(dt, rho, velocity, dij, mass, state, eig) # 核动完了再决定要不要跃迁

            phi21pre = phi2[0]
            phi11pre = phi1[0]
            #phi11s.append(phi1[0])
            #phi12s.append(phi1[1])
            #phi21s.append(phi2[0])
            #phi22s.append(phi2[1])


        if state == 1:
            if position > right_border:
                TU += 1
            else:
                RU += 1
        else:
            if position > right_border:
                TL += 1
            else:
                RL += 1

    end_time = time.time()
    print(f"p0: {p0}, TL_prob: {TL/num_traj}, RU_prob: {RU/num_traj}, RL_prob: {RL/num_traj}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

    TL_probs.append(TL / num_traj)
    RU_probs.append(RU / num_traj)
    RL_probs.append(RL / num_traj)
    TU_probs.append(TU / num_traj)

plt.plot(p0_values, TL_probs, label='TL_prob')
plt.plot(p0_values, RL_probs, label='RL_prob')
plt.plot(p0_values, RU_probs, label='RU_prob')
plt.plot(p0_values, TU_probs, label='TU_prob')
plt.xlabel('p0')
plt.ylabel('Probability')
plt.legend()
plt.show()
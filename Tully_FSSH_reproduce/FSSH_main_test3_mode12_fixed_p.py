import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splder
from Potential_and_der_tempsave_origin import SAC_potential, DAC_potential, ECR_potential
from Force_dij import Cal_dij_and_force
from RK4 import RK4, f
from hoping import hoping_and_adjust
import time

# 参数初始化
testing_mode = int(input("Enter 1 for SAC, 2 for DAC, 3 for ECR: "))
left_border, right_border = -10, 10 # atomic units,下面也都是
p0, mass, dt, num_traj = 20, 2000, 0.5, 1 # p0是动量，mass是质量，dt是时间步长，num_traj是系综轨道数
v0 = p0 / mass # 初始速度
pos_0 = left_border # 初始位置
trans, hopp, refle = 0, 0, 0 # 记录透射、跃迁、反射次数

# 开搞
start_time = time.time()
for trajectory in range(num_traj):
    Max_step = 5000000 # 防止循环不收敛
    state = 0 # 最初电子位于基态, state=0是基态，state=1是激发态
    p, velocity, position = p0, v0, pos_0
    if testing_mode == 1:
        V, V_der = SAC_potential(position)
    elif testing_mode == 2:
        V, V_der = DAC_potential(position)
    else:
        V, V_der = ECR_potential(position)
    rho = np.array([[1+0j, 0+0j], [0+0j, 0+0j]]) # 密度矩阵初始化，最初电子位于基态
    eig, dij, force = Cal_dij_and_force(V, V_der)
    
    while position <= right_border and position >= left_border and Max_step > 0:
        Max_step -= 1
        position, velocity, rho[0, 0], rho[0, 1], rho[1, 1] = RK4([position, velocity, rho[0, 0], rho[0, 1], rho[1, 1]], dt, mass, force, dij, state, V) # 更新这一步长演化后的位置、速度、密度矩阵
        rho[1, 0] = rho[0, 1].conjugate()
        if testing_mode == 1: # 更新势能和势能导数
            V, V_der = SAC_potential(position)
        elif testing_mode == 2:
            V, V_der = DAC_potential(position)
        else:
            V, V_der = ECR_potential(position)
        eig, dij, force = Cal_dij_and_force(V, V_der) # 更新下一步演化所受的力，以及dij、本征态势能
        state, velocity, hop = hoping_and_adjust(dt, rho, velocity, dij, mass, state, V) # 核动完了再决定要不要跃迁

    print(f"第{trajectory+1}条轨道演化完毕")
    if state == 1:
        hopp += 1
    elif position > right_border:
        trans += 1
    else:
        refle += 1

end_time = time.time()
print(f"透射数: {trans}, 跃迁数: {hopp}, 反射数: {refle}")
print(f"运行时间: {end_time - start_time:.2f} 秒")


    





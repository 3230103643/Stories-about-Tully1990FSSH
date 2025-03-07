import numpy as np
import matplotlib.pyplot as plt
from Potential_and_der_tempsave_origin import SAC_potential, DAC_potential, ECR_potential
from Force_dij import Cal_dij_and_force
from RK4 import RK4
from hoping import hoping_and_adjust

# 参数初始化
testing_mode = 2 #int(input("Enter 1 for SAC, 2 for DAC, 3 for ECR: "))#
left_border, right_border = -10, 10 # atomic units,下面也都是
p0, mass, dt, num_traj = 20, 2000, 0.5, 1 # p0是动量，mass是质量，dt是时间步长，num_traj是系综轨道数
v0 = p0 / mass # 初始速度
pos_0 = left_border # 初始位置

# 开搞
for trajectory in range(num_traj):
    print(trajectory)
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
    eig, dij, force, vector = Cal_dij_and_force(V, V_der) # vector即对角化解出来的本征矢矩阵
    totE_initial = 0.5 * mass * v0 ** 2 + eig[0] # 初始总能量 # v0项是核动能，eig[0]是电子哈密顿量的本征值
    
    # 记录步长和state
    steps = []
    states = []
    positions = []
    forces = []
    dijs = []
    rho11s = []
    testrho = []
    phi11s = []
    phi12s = []
    phi21s = []
    phi22s = []
    totEs = []
    eigs = []
    K = []
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
        if phi11s:
            if (phi11s[-1] * phi1[0] < 0) and (abs(phi11s[-1] - phi1[0]) / 2 > 0.05) : # 防止特征向量的正负号突然翻转
                #print(f"phi1: {phi1}")
                reverse_1 = 1
                phi1[0] = -phi1[0]
                phi1[1] = -phi1[1]
                #print(f"phi1: {phi1}") 
        if phi21s:
            if (phi21s[-1] * phi2[0] < 0) and (abs(phi21s[-1] - phi2[0]) / 2 > 0.05) : # 防止特征向量的正负号突然翻转
                reverse_2 = 1
                print(f"phi2: {phi2}") 
                phi2[0] = -phi2[0]
                phi2[1] = -phi2[1]
                #print(f"phi2: {phi2}")
        if reverse_1 + reverse_2 == 1:
            dij[0, 1] = -dij[0, 1]
            dij[1, 0] = -dij[1, 0]
        state, velocity, hop = hoping_and_adjust(dt, rho, velocity, dij, mass, state, eig) # 核动完了再决定要不要跃迁
        #velocity = np.sqrt(2 * (totE_initial - eig[state]) / mass) # 保持总能量不变
        # 记录当前步长和state
        steps.append(5000000-Max_step)
        states.append(state/5) 
        positions.append(position)
        forces.append(force[state])
        dijs.append(dij[0][1])
        rho11s.append(rho[0][0].real)
        testrho.append(rho[0][0].real + rho[1][1].real)
        totEs.append(0.5 * mass * velocity ** 2 + eig[state])
        eigs.append(eig[state])
        K.append(0.5 * mass * velocity ** 2)
        phi11s.append(phi1[0])
        phi12s.append(phi1[1])
        phi21s.append(phi2[0])
        phi22s.append(phi2[1])
    
    # 绘制步长-state图像
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, states, label='states')
    plt.xlabel('step')
    plt.ylabel('states')
    plt.title('Evolution of state over time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rho11s, label='rho11')
    plt.xlabel('step')
    plt.ylabel('rho11')
    plt.title('Evolution of rho11 over time')
    plt.legend()
    plt.grid(True)
    plt.show()

    '''
    fig = plt.figure() # 3D图中绘制特征向量
    ax1 = fig.add_subplot(1,1,1,projection = "3d")
    #x = np.linspace(-10,10,100)
    #y = np.sin(x)
    #z = x
    ax1.plot(steps, phi11s, phi12s, label = "phi1", color = 'blue')
    ax1.plot(steps, phi21s, phi22s, label = "phi2", color = 'red')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('1') 
    ax1.set_zlabel('2')
    ax1.legend()
    plt.show()
    '''

    plt.figure(figsize=(10, 6))
    plt.plot(steps, phi11s, label='phi11', color='blue')
    plt.plot(steps, phi12s, label='phi12', color='red')
    plt.plot(steps, states, label='states', color='green')
    plt.xlabel('step')
    plt.ylabel('phi1')
    plt.title('Evolution of phi1 over time')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, phi21s, label='phi21', color='blue')
    plt.plot(steps, phi22s, label='phi22', color='red')
    plt.plot(steps, states, label='states', color='green')
    plt.xlabel('step')
    plt.ylabel('phi2')
    plt.title('Evolution of phi2 over time')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, forces, label='forces')
    plt.xlabel('step')
    plt.ylabel('forces')
    plt.title('Evolution of force over time')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    plt.figure(figsize=(10, 6))
    plt.plot(steps, positions, label='positions')
    plt.xlabel('step')
    plt.ylabel('positions')
    plt.title('Evolution of position over time')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, dijs, label='dijs')
    plt.xlabel('step')
    plt.ylabel('dijs')
    plt.title('Evolution of d12 over time')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    '''
    plt.figure(figsize=(10, 6))
    plt.plot(steps, testrho, label='rho11+rho22')
    plt.xlabel('step')
    plt.ylabel('rho11+rho22')
    plt.title('Evolution of sum_rho over time')
    plt.legend()
    plt.grid(True)
    plt.show()
    ''' 

    plt.figure(figsize=(10, 6))
    plt.plot(steps, eigs, label='eigen energy', color='blue')
    plt.plot(steps, K, label='kinetic energy', color='red')
    plt.plot(steps, totEs, label='total energy', color='green')
    #plt.plot(steps, states, label='energy', color='black')
    plt.xlabel('step')
    plt.ylabel('energy')
    plt.title('Evolution of eigen energy, kinetic energy, and total energy over time')
    plt.legend()
    plt.grid(True)
    plt.show()
    





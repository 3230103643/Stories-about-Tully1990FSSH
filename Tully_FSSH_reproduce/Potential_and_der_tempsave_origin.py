import numpy as np
import matplotlib.pyplot as plt

def SAC_potential(position): # 计算Simple avoided crossing case 的哈密顿矩阵及其导数
    A,B,C,D = 0.01, 1.6, 0.005, 1.0
    V = np.zeros([2,2])
    V_der = np.zeros([2,2])
    if position > 0:
        V[0,0] = A * (1 - np.exp(-B * position))
        V_der[0,0] = A * B * np.exp(-B * position)
    elif position == 0:
        V[0,0] = 0
        V_der[0,0] = A * B
    else:
        V[0,0] = -A * (1 - np.exp(B * position))
        V_der[0,0] = A * B * np.exp(B * position)
    V[0,1] = C * np.exp(- D * position ** 2)    
    V[1,0] = V[0,1]
    V[1,1] = -V[0,0]
    V_der[0,1] = -2*C*D*position*np.exp(-D*position**2)
    V_der[1,0] = V_der[0,1]
    V_der[1,1] = -V_der[0,0]
    return V, V_der

def DAC_potential(position): # 计算Dual avoided crossing case 的哈密顿矩阵及其导数
    A,B,C,D,E0 = 0.10, 0.28, 0.015, 0.06, 0.05
    V = np.zeros([2,2])
    V_der = np.zeros([2,2])
    V[0,0] = 0
    V[1,1] = -A * np.exp(-B * position ** 2) + E0
    V[0,1] = C * np.exp(-D * position ** 2)
    V[1,0] = V[0,1]
    V_der[1, 1] = A * B * 2 * position * np.exp(-B*position**2)
    V_der[0, 1] = -2 * position * C * D * np.exp( -D*position**2 )
    V_der[1, 0] = V_der[0, 1]
    return V, V_der

def ECR_potential(position): # 计算Extended Coupled avoided crossing case 的哈密顿矩阵及其导数
    A,B,C = 0.0006, 0.10, 0.90
    V = np.zeros([2,2])
    V_der = np.zeros([2,2])
    V[0,0] = A
    V[1,1] = -A
    if position < 0:
        V[0,1] = B * np.exp(C * position)
        V_der[0,1] = B * C * np.exp(C * position)
    elif position == 0:
        V[0,1] = B
        V_der[0,1] = B*C
    else:
        V[0,1] = B * (2 - np.exp(-C * position))
        V_der[0,1] = B * C * np.exp(-C * position)
    V_der[0,0] =0
    V_der[1,1] = 0
    V[1,0] = V[0,1]
    V_der[1,0] = V_der[0,1]
    return V, V_der

def main():

	npoints = 200
	PMin = -10
	PMax = 10

	plist = np.linspace(PMin, PMax, npoints) # 坐标分割/横坐标初始化

	# SAC
    # eig, vector = linalg.eigh()
	eig_SAC = np.array([ np.linalg.eigh(SAC_potential(p)[0])[0] for p in plist ]) # 对于plist中的每一个p，计算其势能矩阵的本征值
	plt.plot(plist, eig_SAC, linewidth=2, linestyle='-', color='red') # 横轴p，纵轴每个p对应的势能矩阵的本征值，绘图
	plt.show()
	
	# DAC
	eig_DAC = np.array([ np.linalg.eigh(DAC_potential(p)[0])[0] for p in plist ])
	plt.plot(plist, eig_DAC, linewidth=2, linestyle='-', color='blue')
	plt.show()

	# ECR
	eig_ECR = np.array([ np.linalg.eigh(ECR_potential(p)[0])[0] for p in plist ])
	plt.plot(plist, eig_ECR, linewidth=2, linestyle='-', color='pink')
	plt.show()	

if __name__ == "__main__":

	main()

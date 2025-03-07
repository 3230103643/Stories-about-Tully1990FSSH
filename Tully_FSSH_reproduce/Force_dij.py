import numpy as np
from scipy.interpolate import splev, splrep, splder
import matplotlib.pyplot as plt
from Potential_and_der_tempsave_origin import SAC_potential, DAC_potential, ECR_potential

def Cal_dij_and_force(V, V_der): # 计算非绝热耦合和核受到的力
    eig, vector = np.linalg.eigh(V) # eig:含两个元素的数组，eig[0]基态本征能量，eig[1]激发态能量；vector：2x2矩阵，每一列是一个本征态的基组展开系数
    idx = eig.argsort() # 确保本征态能量升序排列
    eig = eig[idx]
    vector = vector[:, idx]
    dij = np.zeros([2,2])
    force = np.zeros([2]) # 这里力是一个二维向量，表示不同电子态下核受到的力
    #print("vector:",vector)
    #phi1 = vector[:, 0]
    #phi2 = vector[:, 1]
    #F1 = phi1.T @ V_der @ phi1
    #F2 = phi2.T @ V_der @ phi2
    #force = np.array([F1, F2])
    MatrixDot = np.dot( np.dot( vector.T, V_der ), vector) # 这里的代数部分没有问题
    pot_gap = eig[1] - eig[0] # 两个绝热态之间的势能差
    dij[0, 1] = (MatrixDot[0][1]) / pot_gap
    #dij[0, 1] = (phi1.T @ V_der @ phi2) / pot_gap
    dij[1, 0] = -1.0 * dij[0, 1]
    force[0] = -1.0 * MatrixDot[0][0]
    force[1] = -1.0 * MatrixDot[1][1]
    return eig, dij, force, vector

def Plot(type, plist):

	factor = 1

	if type == "ECR":
		V = np.array([ECR_potential(p)[0] for p in plist])
		V_der = np.array([ECR_potential(p)[1] for p in plist])
		factor = 1
	if type == "SAC":
		V = np.array([SAC_potential(p)[0] for p in plist])
		V_der = np.array([SAC_potential(p)[1] for p in plist])
		factor = 50
	if type == "DAC":
		V = np.array([DAC_potential(p)[0] for p in plist])
		V_der = np.array([DAC_potential(p)[1] for p in plist])
		factor = 15

	eig = np.array([ Cal_dij_and_force(V=V[position], V_der=V_der[position])[0] for position in range(len(plist))])
	dij = np.array([ Cal_dij_and_force(V=V[position], V_der=V_der[position])[1] for position in range(len(plist))])
	force = np.array([ Cal_dij_and_force(V=V[position], V_der=V_der[position])[2] for position in range(len(plist))])

	# dij
	plt.plot(plist, dij[:, 0, 1]/factor, linewidth=2, linestyle='-', color='black')
	# force
	plt.plot(plist, force[:, 0], linewidth=2, linestyle='-', color='red')
	#plt.plot(plist, force[:, 1], linewidth=2, linestyle=':', color='red')
	# eig
	plt.plot(plist, eig[:, 0], linewidth=2, linestyle='-', color='blue')
	plt.plot(plist, eig[:, 1], linewidth=2, linestyle=':', color='blue')
	
	plt.legend(["dij/%d" % factor, 'force1', 'force2', 'eig1', 'eig2'], loc=1, fontsize='large')

	plt.show()	
	
def main():

	npoints = 100
	min    = -10
	max    = 10
	plist = np.linspace(min, max, npoints)

	type = "DAC" # ECR, SAC, DAC

	# plot	
	# dij of factor = 50 for SAC, 12 for DAC, 1 for ECR 
	Plot(type="SAC", plist=plist)
	Plot(type="DAC", plist=plist)
	Plot(type="ECR", plist=plist)

if __name__ == "__main__":
	main()
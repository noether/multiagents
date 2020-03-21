from scipy import linalg as la
from scipy.integrate import odeint
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

# Happy pdf for a happy submission without complains in paperplaza, arxiv, etc
font = {'size'   : 20}

matplotlib.rc('font', **font)

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

# Shape
m = 2
p_star_square = np.array([0,0,0,10,10,10,10,0])
p_star_square_nine = np.array([0,0,0,10,0,20,10,20,10,10,10,0,20,0,20,10,20,20])

p_star = p_star_square_nine

# Network
def build_B(edges, numagents):
    B = np.zeros((numagents, len(edges)))
    for n,e in enumerate(edges):
        i = e[0]-1
        j = e[1]-1
        B[i][n] = 1 
        B[j][n] = -1
    return B

B_square = np.array([[1, 0, 1],[-1, 1, 0],[0, -1, 0],[0, 0, -1]])

list_edges_nine = ((1,2),(2,3),(3,4),(4,5),(5,6),(5,8),(8,7),(8,9))
B_square_nine = build_B(list_edges_nine,9)

B = B_square_nine
numAgents, numEdges = B.shape

L = B.dot(B.T)

# Designing desired translation
M_square = np.array([[0, 0, 1],[0, 1, 0],[0, 1, 0],[0, 0, 1]])

if numAgents == 9:
    M_square_nine = np.zeros_like(B)
    M_square_nine[1-1][1-1] = 1
    M_square_nine[2-1][1-1] = 1
    M_square_nine[3-1][2-1] = 1
    M_square_nine[4-1][4-1] = -1
    M_square_nine[5-1][4-1] = -1
    M_square_nine[6-1][5-1] = -1
    M_square_nine[7-1][7-1] = -1
    M_square_nine[8-1][8-1] = 1
    M_square_nine[9-1][8-1] = 1

M = M_square_nine
Lambda = M.dot(B.T)
kappa = 0.1

# Wrong measurements
scaling_factors = np.random.uniform(0.95,1.05, numAgents)

Da = np.diag(scaling_factors)

def build_DR_2D(misalignments):
    for i in range(len(misalignments)):
        phi = misalignments[i]
        R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        if i == 0:
            DR_2D = R
        else:
            DR_2D = la.block_diag(DR_2D, R)
    return DR_2D

misalignments = np.random.uniform(-0.17,0.17, numAgents)

DR = build_DR_2D(misalignments)

# Kronecker products
Bb = la.kron(B, np.eye(m))
Lb = la.kron(L, np.eye(m))
Mb = la.kron(M, np.eye(m))
Lambdab = la.kron(Lambda, np.eye(m))
Dab = la.kron(Da, np.eye(m))

Dx = Dab.dot(DR)

# Initial conditions
p0 = np.random.uniform(-20,20, numAgents*m)

# ODE(s)
def displacement_based(p, t, Lb, p_star):
    dpdt = -Lb.dot(p-p_star)
    return dpdt

def displacement_based_maneuvering(p, t, Lb, Lambdab, p_star):
    dpdt = -(Lb - kappa*Lambdab).dot(p) + Lb.dot(p_star)
    return dpdt

def displacement_based_scale_factor(p, t, Lb, Dab, p_star):
    dpdt = -(Dab.dot(Lb)).dot(p) + Lb.dot(p_star)
    return dpdt

def displacement_based_scale_factor_and_miss(p, t, Lb, Dx, p_star):
    dpdt = -(Dx.dot(Lb)).dot(p) + Lb.dot(p_star)
    return dpdt

# Simulation
t = np.linspace(0, 100, 100001)
#p = odeint(displacement_based, p0, t, args=(Lb,p_star))
#p = odeint(displacement_based_maneuvering, p0, t, args=(Lb,Lambdab,p_star))
#p = odeint(displacement_based_scale_factor, p0, t, args=(Lb,Dab,p_star))
p = odeint(displacement_based_scale_factor_and_miss, p0, t, args=(Lb,Dx,p_star))

# Post processing
agentcolor = ['r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g']
plt.close("all")
plt.ion()
plt.figure(0)

def plot_edges(fig, X, B, m, n):
    agents, edges = B.shape
    f = plt.figure(fig)
    a, b = 0, 0

    for i in range(0, edges):
        for j in range(0, agents):
            if B[j,i] == 1:
                a = j
            elif B[j,i] == -1:
                b = j

        if m == 2:
            if i > 9:
                plt.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], 'r--', lw=1.5)
            else:
                plt.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], 'k--', lw=1.5)

        if m == 3:
            ax = f.gca(projection='3d')

            if i == n:
                ax.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], [X[m*a+2], X[m*b+2]], 'r--', lw=1.5)
            else:
                ax.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], [X[m*a+2], X[m*b+2]], 'k--', lw=1.5)

if m == 2:
    for i in range(numAgents):
        plt.plot(p[:, m*i],p[:, (m*i)+1], agentcolor[i])
        plt.plot(p[0, m*i],p[0, (m*i)+1], 'x'+agentcolor[i], mew=1)
        plt.plot(p[-1, m*i],p[-1, (m*i)+1], 'o'+agentcolor[i], mew=1)
        plot_edges(0, p[-1,:], B, m, -1)

    plt.grid()
    plt.axis("equal")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if m == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(numAgents):
        ax.plot(p[:,m*i],p[:,(m*i)+1],p[:,(m*i)+2], c=agentcolor[i])
        ax.scatter(p[0,m*i],p[0,(m*i)+1],p[0,(m*i)+2], c=agentcolor[i], marker='x')
        ax.scatter(p[-1,m*i],p[-1,(m*i)+1],p[-1,(m*i)+2], c=agentcolor[i])
    plt.show()

z_star = Bb.T.dot(p_star)
z_tilde_star = la.inv(Bb.T.dot(Dx).dot(Bb)).dot(Bb.T).dot(Bb).dot(z_star)
M_breve = -(Dx.dot(Bb) - Bb.dot(la.inv(Bb.T.dot(Bb))).dot(Bb.T).dot(Dx).dot(Bb))
M_breve_zstar = Bb - Dx.dot(Bb).dot(la.inv(Bb.T.dot(Dx).dot(Bb))).dot(Bb.T).dot(Bb)
v_star = M_breve_zstar.dot(z_star)


e_distortion = np.zeros(len(t))
v_distortion = np.zeros(len(t))
for i,time in enumerate(t):
    e_distortion[i] = la.norm(z_tilde_star -  Bb.T.dot(p[i,:]))
    v_distortion[i] = la.norm(v_star -  (-(Dx.dot(Lb)).dot(p[i,:]) + Lb.dot(p_star)))

plt.figure(1)
plt.plot(t, e_distortion, label='$||z(t) - \\tilde z||$')
plt.plot(t, v_distortion, label='$||\\frac{\\mathrm{d}p(t)}{\\mathrm{dt}} - (\\mathbf{1}_n \otimes v^*)||$')
plt.legend()
plt.grid()
plt.xlim(0,25)
plt.xlabel('Time')

z_predicted_error = z_tilde_star - Bb.T.dot(p[-1,:])
print(z_predicted_error)

v_predicted_error = v_star - (-(Dx.dot(Lb)).dot(p[-1,:]) + Lb.dot(p_star))
print(v_predicted_error)

plt.pause(0)

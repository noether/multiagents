from scipy import linalg as la
from scipy.integrate import odeint
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

import picos as pic

np.core.arrayprint._line_width = 150

# Happy pdf for a happy submission without complains in paperplaza, arxiv, etc
font = {'size'   : 20}

matplotlib.rc('font', **font)

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True



def build_B(list_edges, n):
    B = np.zeros((n,len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0]-1, i] = 1
        B[list_edges[i][1]-1, i] = -1
    return B

def build_Laplacian(B, w):
    L = B.dot(np.diag(w).dot(B.T))
    return L

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
            if i > n:
                plt.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], 'r--', lw=1.5)
            else:
                plt.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], 'k--', lw=1.5)

        if m == 3:
            ax = f.gca(projection='3d')

            if i == n:
                ax.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], [X[m*a+2], X[m*b+2]], 'r--', lw=1.5)
            else:
                ax.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], [X[m*a+2], X[m*b+2]], 'k--', lw=1.5)

def find_w_from_ps(ps, B, m):
    # Algorithm from "Affine Formation Maneuver Control of Multiagent Systems"
    # Transactions on Automatic Control 2017
    # Author: Shiyu Zhao
    numAgents = B.shape[0]

    P = ps.reshape(numAgents,m)
    Pbar = np.concatenate((P.T,np.ones(numAgents).T),axis=None)
    Pbar = Pbar.reshape(m+1,numAgents).T

    H = B.T
    
    E = Pbar.T.dot(H.T).dot(np.diag(H[:,0]))

    for i in range(1,H.shape[1]):
        aux = Pbar.T.dot(H.T).dot(np.diag(H[:,i]))
        E = np.concatenate((E,aux))

    ker_E = la.null_space(E)

    [U,s,Vh] = la.svd(Pbar)

    U2 = U[:,m+1:]

    M = []
    for i in range(ker_E.shape[1]):
        aux = U2.T.dot(H.T).dot(np.diag(ker_E[:,i])).dot(H).dot(U2)
        M.append(aux)

    Mc = pic.new_param('Mc',M)
    lmi_problem = pic.Problem()
    c = lmi_problem.add_variable("c", len(M))
    lmi_problem.add_constraint(pic.sum([c[i]*Mc[i] for i in range(len(M))]) >> 0)
    lmi_problem.set_objective('find',c)
    lmi_problem.solve(verbose = 0, solver='smcp')

    w = np.zeros(ker_E.shape[0])
    for i in range(ker_E.shape[1]):
        w = w + (c[i].value * ker_E[:,i])

    return w

# ODE
def affine_formation_control(p, t, L):
    dpdt = -L.dot(p)

    return dpdt

if __name__ == "__main__":
    # Shape
    m = 2
    #ps = np.array([2,0,1,1,1,-1,0,1,0,-1,-1,1,-1,-1])
    ps = np.array([2,0,1,1,1,-1,0,1,0,-1,-1,1,-1,-1,-2,0])
    numAgents = len(ps)/m
    #list_edges = ((1,2),(1,3),(1,4),(1,5),(2,4),(2,7),(3,5),(3,6),(4,5),(4,6),(5,7),(6,7))
    list_edges = ((1,2),(1,3),(1,4),(1,5),(2,4),(2,7),(3,5),(3,6),(4,5),(4,6),(5,7),(6,8),(7,8),(4,8),(5,8))
    B = build_B(list_edges, numAgents)
    w = find_w_from_ps(ps, B, m)
    L = build_Laplacian(B,w)
    Bb = la.kron(B, np.eye(m))
    Lb = la.kron(L, np.eye(m))

    Mr = np.zeros_like(B)
    Mr[0,0] = -0.5
    Mr[0,1] = 0.5
    Mr[1,0] = -1/np.sqrt(2)
    Mr[2,1] = 1/np.sqrt(2)
    Mr[3,4] = -1
    Mr[4,6] = 1
    Mr[5,11] = -1/np.sqrt(2)
    Mr[6,12] = 1/np.sqrt(2)
    Mr[7,11] = -0.5
    Mr[7,12] = 0.5
    kr = 0.0

    Mt = np.zeros_like(B)
    Mt[0,0] = 0.5
    Mt[0,1] = 0.5
    Mt[1,4] = 1
    Mt[2,6] = 1
    Mt[3,4] = 1
    Mt[4,6] = 1
    Mt[5,9] = 1
    Mt[6,10] = 1
    Mt[7,11] = 0.5
    Mt[7,12] = 0.5
    kt = 0.0

    Mh = np.zeros_like(B)
    Mh[1,4] = 1
    Mh[2,6] = -1
    Mh[3,4] = 1
    Mh[4,6] = -1
    Mh[5,9] = 1
    Mh[6,10] = -1
    kh = 0.01

    Ms = np.zeros_like(B)
    Ms[0,0] = 0.5
    Ms[0,1] = 0.5
    Ms[1,2] = 0.5/np.sqrt(2)
    Ms[2,7] = 0.5/np.sqrt(2)
    Ms[3,8] = 0.5
    Ms[4,8] = -0.5
    Ms[5,7] = -0.5/np.sqrt(2)
    Ms[6,2] = -0.5/np.sqrt(2)
    Ms[7,11] = -0.5
    Ms[7,12] = -0.5
    ks = 0

    Mrb = la.kron(Mr, np.eye(m))
    Mtb = la.kron(Mt, np.eye(m))
    Mhb = la.kron(Mh, np.eye(m))
    Msb = la.kron(Ms, np.eye(m))

    Lmodb = Lb -kr*Mrb.dot(Bb.T) -kt*Mtb.dot(Bb.T) -kh*Mhb.dot(Bb.T) -ks*Msb.dot(Bb.T)

    # Simulation
    t = np.linspace(0, 350, 80001)
    p0 = np.random.uniform(-10,10, numAgents*m)
    p = odeint(affine_formation_control, p0, t, args=(Lmodb,))

    # Post processing
    agentcolor = ['r', 'g', 'b', 'k', 'm', 'c', 'g','r']
    plt.figure(0)
    plt.clf()

    if m == 2:
        for i in range(numAgents):
            plt.plot(p[:, m*i],p[:, (m*i)+1], agentcolor[i])
            plt.plot(p[0, m*i],p[0, (m*i)+1], 'x'+agentcolor[i], mew=2)
            plt.plot(p[-1, m*i],p[-1, (m*i)+1], 'o'+agentcolor[i], mew=2)

        plot_edges(0, p[-1,:], B, m, 15)
        #plot_edges(0, p[-20000,:], B, m, 15)
        plt.axis("equal")
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("Agents' trajectory")

        #a = plt.axes([0.20,0.5,0.20,0.30])
        #xavg = sum(p[-1,range(0,numAgents*2,2)])/numAgents
        #yavg = sum(p[-1,range(1,numAgents*2,2)])/numAgents
        #a.set_xlim(xavg-0.15,xavg+0.15)
        #a.set_ylim(yavg-0.15,yavg+0.15)
        #for i in range(numAgents):
        #    a.plot(p[:, m*i],p[:, (m*i)+1], agentcolor[i])
        #    a.plot(p[0, m*i],p[0, (m*i)+1], 'x'+agentcolor[i], mew=2)
        #    a.plot(p[-1, m*i],p[-1, (m*i)+1], 'o'+agentcolor[i], mew=2)

        #plot_edges(0, p[-1,:], B, m, 15)
        #a.grid(True)

        plt.show()

    if m == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(numAgents):
            ax.plot(p[:,m*i],p[:,(m*i)+1],p[:,(m*i)+2], c=agentcolor[i])
            ax.scatter(p[0,m*i],p[0,(m*i)+1],p[0,(m*i)+2], c=agentcolor[i], marker='x')
            ax.scatter(p[-1,m*i],p[-1,(m*i)+1],p[-1,(m*i)+2], c=agentcolor[i])
        plt.show()


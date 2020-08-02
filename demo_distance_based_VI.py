import pygame
import matplotlib.pyplot as pl
import drawmisc
import agents as ag
import numpy as np
import logpostpro as lp

# setup simulation
WHITE = (255, 255, 255)

# Desired configuration
# desired_configuration = [(0,0), (10,0), (10,10), (0,10)] # Square

desired_configuration = [(0,0), (10,5), (5,12), (-15,10), (-15,0), (-5, 15)]
markers = ['^','>','v','<','1','2'] # At least same number as the num of agents

undirected_edges = [(0,1), (1,2), (2,3), (3,0), (1,3), (0,4), (1,4), (3,5), (2,5)]
listofedges_and_distances = []

for edge in undirected_edges:
    pi = np.asarray(desired_configuration[edge[0]])
    pj = np.asarray(desired_configuration[edge[1]])
    d = np.linalg.norm(pi-pj)
    listofedges_and_distances.append((edge[0],edge[1],d))

# The final shape must be congruent with the desired one
complete_graph_edges = []
for i in range(0,len(desired_configuration)):
    for j in range(i,len(desired_configuration)):
        if(i != j):
            complete_graph_edges.append((i,j))

numagents = len(desired_configuration)

# Incidence matrix
B = np.zeros((numagents, len(listofedges_and_distances)))

for idx,edge in enumerate(listofedges_and_distances):
    B[edge[0],idx] =  1
    B[edge[1],idx] = -1

# Simulation
dt = 1e-2
num_steps = 400

cc = 0 # number of non final congruent

# Figures
fig0 = pl.figure(0)
ax0 = fig0.add_subplot(111)

plot_trajectories = 0
listoffigs = []

colors = pl.rcParams['axes.prop_cycle'].by_key()['color']

for idx,pos in enumerate(desired_configuration):
    pd = np.asarray(pos)
    ax0.plot(pd[0],pd[1], 'o', color=colors[idx])

agents, edges = B.shape
a, b = 0, 0
X = np.array(desired_configuration)
X = X.reshape(2*agents,1)

for i in range(0, edges):
    for j in range(0, agents):
        if B[j,i] == 1:
            a = j
        elif B[j,i] == -1:
            b = j
    ax0.plot([X[2*a], X[2*b]], [X[2*a+1], X[2*b+1]], 'k--', lw=1.5)

limitsarea = 800
num_simulations = 10000

selected_agent = 1

buscar = 400.0

for num_sim in range(1,num_simulations+1):

    listofagents = []

    for i in range(numagents):
        if i == selected_agent:
            init_pos = np.array(desired_configuration[i]) + (buscar*np.random.rand(2) - buscar/2.0)
            listofagents.append(ag.AgentDI(WHITE, i, init_pos, 0*np.random.rand(2)))
        else:
            listofagents.append(ag.AgentDI(WHITE, i, np.array(desired_configuration[i]), 0*np.random.rand(2)))

    for agent in listofagents:
        agent.distance_based_kv = 5

    for step in range(num_steps-1):

        # Lists of neighbors in an undirected graph (passed by copy... not very efficient)
        # It would be great to pass them (only once) before the simulation by reference/"pointer"
        for edge in listofedges_and_distances:
            listofagents[edge[0]].neighbors.append(listofagents[edge[1]])
            listofagents[edge[0]].desired_distances.append(edge[2])
            listofagents[edge[1]].neighbors.append(listofagents[edge[0]])
            listofagents[edge[1]].desired_distances.append(edge[2])

        # Execute the consensus algorithm in a distributed way
        for agent in listofagents:
            agent.distance_based_VI(dt)
            agent.neighbors = []

    # Postprocessing
    congruent = 1
    for edge in complete_graph_edges:
        pi = np.asarray(listofagents[edge[0]].pos)
        pj = np.asarray(listofagents[edge[1]].pos)
        d = np.linalg.norm(pi-pj)

        pic = np.asarray(desired_configuration[edge[0]])
        pjc = np.asarray(desired_configuration[edge[1]])
        dc = np.linalg.norm(pic-pjc)

        if(np.abs((d-dc)) > 1):
            congruent = 0
        if(np.linalg.norm(pi) > 1e4):
            congruent = 0
        if(np.isnan(np.linalg.norm(pi))):
            congruent = 0

    if(congruent):
        ax0.plot(init_pos[0],init_pos[1], 'x', color=colors[selected_agent])

        cc = cc + 1
        print("Congruent number", cc, " Num sim: ", num_sim)

ax0.axis("equal")
fig0.show()

#for figure in listoffigs:
#    figure.show()

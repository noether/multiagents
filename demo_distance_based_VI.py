import pygame
import matplotlib.pyplot as pl
import drawmisc
import agents as ag
import numpy as np
import logpostpro as lp

# setup simulation
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

# Desired configuration
desired_configuration = [(0,0), (10,0), (10,10), (0,10)]
undirected_edges = [(0,1), (1,2), (2,3), (3,0), (1,3), (0,2)]
listofedges_and_distances = []

for edge in undirected_edges:
    pi = np.asarray(desired_configuration[edge[0]])
    pj = np.asarray(desired_configuration[edge[1]])
    d = np.linalg.norm(pi-pj)
    listofedges_and_distances.append((edge[0],edge[1],d))

numagents = len(desired_configuration)
listofagents = []

for i in range(numagents):
    listofagents.append(ag.AgentDI(WHITE, i, 5*np.random.rand(2), 0*np.random.rand(2)))

for agent in listofagents:
    agent.traj_draw = False

# Incidence matrix
B = np.zeros((numagents, len(listofedges_and_distances)))

for idx,edge in enumerate(listofedges_and_distances):
    B[edge[0],idx] =  1
    B[edge[1],idx] = -1

# Simulation
dt = 5e-3
num_steps = 400

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

# Translate and rotate to plot vs desired configuration
p_pretty = []

p1d = np.asarray(desired_configuration[0])
p2d = np.asarray(desired_configuration[1])
thetad = np.arctan2((p2d-p1d)[1],(p2d-p1d)[0])

p1 = listofagents[0].pos
p2 = listofagents[1].pos
theta = np.arctan2((p2-p1)[1],(p2-p1)[0])

st = np.sin(theta - thetad)
ct = np.cos(theta - thetad)
Rot = np.array([[ct, st],[-st,ct]])

for agent in listofagents:
    p_pretty.append(Rot.dot(agent.pos - listofagents[0].pos))

fig = pl.figure(0)
ax = fig.add_subplot(111)
lp.plot_trajectories(ax, listofagents, B)
ax.axis("equal")
pl.show()

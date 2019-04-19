import pygame
import matplotlib.pyplot as pl
import drawmisc
import agents as ag
import numpy as np
import logpostpro as lp

# setup simulation
WIDTH = 1000
HEIGHT = 1000

CENTERX = WIDTH/2
CENTERY = WIDTH/2

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

size = [WIDTH, HEIGHT]
screen = pygame.display.set_mode(size)

numagents = 4
listofagents = []

# Desired configuration
desired_configuration = [(0,0), (100,0), (0,100), (100,100)]
undirected_edges = [(0,1), (1,2), (2,3), (0,3)]
listofedges_and_distances = []

for edge in undirected_edges:
    pi = np.asarray(desired_configuration[edge[0]])
    pj = np.asarray(desired_configuration[edge[1]])
    d = np.linalg.norm(pi-pj)
    listofedges_and_distances.append((edge[0],edge[1],d))

for i in range(numagents):
    listofagents.append(ag.AgentDI(WHITE, i, 1000*np.random.rand(2), 50-100*np.random.rand(2)))

for agent in listofagents:
    agent.traj_draw = False

# Incidence matrix
B = np.zeros((numagents, len(listofedges_and_distances)))

for idx,edge in enumerate(listofedges_and_distances):
    B[edge[0],idx] =  1
    B[edge[1],idx] = -1

# run simulation
pygame.init()
clock = pygame.time.Clock()
fps = 50
dt = 1.0/fps
time = 0

runsim = True
while(runsim):
    screen.fill(BLACK)

    # Lists of neighbors in an undirected graph (passed by copy... not very efficient)
    # It would be great to pass them (only once) before the simulation by reference/"pointer"
    for edge in listofedges_and_distances:
        listofagents[edge[0]].neighbors.append(listofagents[edge[1]])
        listofagents[edge[0]].desired_distances.append(edge[2])
        listofagents[edge[1]].neighbors.append(listofagents[edge[0]])
        listofagents[edge[1]].desired_distances.append(edge[2])

    # Execute the consensus algorithm in a distributed way
    for agent in listofagents:
        agent.draw(screen)
        agent.distance_based(dt)
        agent.neighbors = []

    # Draw the connected agents with a dashed line
    for edge in listofedges_and_distances:
        drawmisc.draw_dashed_line(screen, WHITE, (listofagents[edge[0]].pos[0],HEIGHT-listofagents[edge[0]].pos[1]), (listofagents[edge[1]].pos[0],HEIGHT-listofagents[edge[1]].pos[1]))

    clock.tick(fps)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            endtime = pygame.time.get_ticks()
            pygame.quit()
            runsim = False


# Postprocessing
fig = pl.figure(0)
ax = fig.add_subplot(111)
lp.plot_trajectories(ax, listofagents)
ax.axis("equal")
pl.show()

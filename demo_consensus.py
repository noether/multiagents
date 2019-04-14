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

for i in range(numagents):
    listofagents.append(ag.AgentDI(WHITE, i, 1000*np.random.rand(2), 50-100*np.random.rand(2)))

for agent in listofagents:
    agent.traj_draw = False

listofedges = [(0,1), (1,2), (2,3), (0,3)]

# Incidence matrix
B = np.zeros((numagents, len(listofedges)))

for idx,edge in enumerate(listofedges):
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
    for edge in listofedges:
        listofagents[edge[0]].neighbors.append(listofagents[edge[1]])
        listofagents[edge[1]].neighbors.append(listofagents[edge[0]])

    # Execute the consensus algorithm in a distributed way
    for agent in listofagents:
        agent.draw(screen)
        agent.consensus(dt)

    # Draw the connected agents with a dashed line
    for edge in listofedges:
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
for agent in listofagents:
    lp.plot_position(ax, agent)
ax.axis("equal")
pl.show()

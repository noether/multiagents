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

agentsi = ag.AgentSI(WHITE, 1, np.array([500.0,500.0]))
agentdi = ag.AgentDI(RED, 2, np.array([240.0,250.0]), np.array([0.0, 6.0]))
agentuni = ag.AgentUnicycle(GREEN, 3, np.array([600.0, 600.0]), np.array([5.0,0.0]))

# run simulation
pygame.init()
clock = pygame.time.Clock()
fps = 50
dt = 1.0/fps
time = 0

runsim = True
while(runsim):
    screen.fill(BLACK)
    agentsi.draw(screen)
    agentdi.draw(screen)
    agentuni.draw(screen)
    drawmisc.draw_dashed_line(screen, WHITE, (0,screen.get_height()-0), (100,screen.get_height()-100))
    agentsi.step_dt(np.array([30,2]), dt)
    agentdi.step_dt(np.array([0,0]), dt)
    agentuni.step_dt(0, 0, dt)
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
lp.plot_position(ax, agentsi)
lp.plot_position(ax, agentdi)
lp.plot_position(ax, agentuni)
ax.axis("equal")
pl.show()

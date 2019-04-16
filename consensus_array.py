import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Global options
np.random.seed(10)
do_debug = False
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Model parameters
num_agents = 5
space_dim  = 2
num_steps  = 100
step_size  = 0.9e-1

# Parameters for consensus
consensus_kc = 1e-1
consensus_kv = 1

# Generate arrays and data
times = np.arange(num_steps)

x = np.zeros((num_agents, num_steps, space_dim+1)) # agent, step, (t, x, y, ...)
v = np.zeros((num_agents, num_steps, space_dim+1)) # agent, step, (vt,vx,vy, ...)

# Random initial positions and velocities:
a = -1
b = +1
x[:,0,1:space_dim+1] = (b - a) * np.random.random((num_agents, space_dim)) + a
v[:,0,1:space_dim+1] = (b - a) * np.random.random((num_agents, space_dim)) + a

# Generate evolution for each time step (s) for each agent (j)
for step in range(num_steps-1):
    for j in range(num_agents):
        u0 = np.sum(x[j, step, 1:space_dim+1] - x[:, step, 1:space_dim+1], axis=0)
        u = -consensus_kc*u0 -consensus_kv*v[j, step, 1:space_dim+1]
        v[j, step+1, 1:space_dim+1] = v[j, step, 1:space_dim+1] + u*step_size
        x[j, step+1, 1:space_dim+1] = x[j, step, 1:space_dim+1] + \
                                      v[j, step, 1:space_dim+1]*step_size
        x[j, step+1, 0] = step+1
        times[step] = step
        if do_debug:
            if j == 0:
                print(u0, u, x, v)

# Compute distance from converging point and speed arrays
final_pos = np.average(x[:,-1,1:space_dim+1], axis=(0))
dista = np.linalg.norm(x[:,:,1:space_dim+1] - final_pos, axis=(2)) # size = num_agents x num_steps
speed = np.linalg.norm(v[:,:,1:space_dim+1], axis=(2)) # size = num_agents x num_steps


# Plot variables
fig = plt.figure(figsize=(20,8))
gs = gridspec.GridSpec(2, 5)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax2a = fig.add_subplot(gs[0, 2])
ax2b = fig.add_subplot(gs[1, 2])
ax3a = fig.add_subplot(gs[0, 3])
ax3b = fig.add_subplot(gs[1, 3])
ax4a = fig.add_subplot(gs[0, 4])
ax4b = fig.add_subplot(gs[1, 4])

for j in range(num_agents):
    ax1.plot(x[j,:,1], x[j,:,2], marker='.', ls='-',  color=colors[j])
    ax2a.plot(times, x[j,:,1], marker='.', ls='-',  color=colors[j])
    ax2b.plot(times, x[j,:,2], marker='.', ls='-',  color=colors[j])
    ax3a.plot(times, v[j,:,1], marker='.', ls='-',  color=colors[j])
    ax3b.plot(times, v[j,:,2], marker='.', ls='-',  color=colors[j])
    ax4a.plot(times, speed[j], marker='', ls='-',  color=colors[j])
    ax4b.plot(times, dista[j], marker='', ls='-',  color=colors[j])

# Plot average speed and distance
ax4a.plot(times, np.average(speed[:,:], axis=0), marker='', ls='-',
          lw=3, color='k', label='average speed')
ax4b.plot(times, np.average(dista[:,:], axis=0), marker='', ls='-',
          lw=3, color='k', label='average distance')
ax4a.legend(loc=0)
ax4b.legend(loc=0)


# Labels etc
ax1.set_title('X-Y')
ax2a.set_title('X vs time')
ax2b.set_title('Y vs time')
ax3a.set_title('Vx vs time')
ax3b.set_title('Vy vs time')
ax4a.set_title('Speed vs time')
ax4b.set_title('Distance vs time')
ax1.set_aspect('equal')
ax4a.semilogy()
ax4b.semilogy()

fig.savefig('test1.png', bbox_inches='tight')



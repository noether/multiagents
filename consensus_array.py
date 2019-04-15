import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


np.random.seed(10)

do_debug = False

num_agents = 5
num_times = 100

data = np.zeros((num_times, num_agents, 5))   # [t, agent, x, y, vx, vy, t]
                                              #  0  1      2  3   4   5  6
                                              #            0  1   2   3  4

# Generate positions and velocities in the range [-1,1]
a = -1
b = +1
data[0,:,0:4] = (b - a) * np.random.random((num_agents, 4)) + a

# Parameters for consensus
consensus_kc = 1e-1
consensus_kv = 1
dt = 2e-1
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Generate evolution with time (t) and agent (j)
for t in range(num_times-1):
    for j in range(num_agents):
        u0 = np.sum(data[t, j, 0:2]- data[t, :, 0:2], axis=0)
        u = -consensus_kc*u0 -consensus_kv*data[t, j, 2:4]
        data[t+1, j, 2:4] = data[t, j, 2:4] + u*dt
        data[t+1, j, 0:2] = data[t, j, 0:2] + data[t, j, 2:4]*dt
        data[t+1, j, 4] = t+1
        if do_debug:
            if j == 0:
                print(u0, u, data[t,j,0], data[t,j,1], data[t,j,2],
                             data[t,j,3], data[t,j,4])

# Compute distance from converging point and speed arrays
final_pos = np.average(data[-1,:,0:2], axis=(0))
dista = np.linalg.norm(data[:,:,0:2]-final_pos, axis=(2))
speed = np.linalg.norm(data[:,:,2:4], axis=(2))

# Save the numpy array
np.save('data', data)


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
    ax1.plot(data[:,j,0], data[:,j,1], marker='.', ls='-',  color=colors[j])
    ax2a.plot(data[:,j,4], data[:,j,0], marker='.', ls='-',  color=colors[j])
    ax2b.plot(data[:,j,4], data[:,j,1], marker='.', ls='-',  color=colors[j])
    ax3a.plot(data[:,j,4], data[:,j,2], marker='.', ls='-',  color=colors[j])
    ax3b.plot(data[:,j,4], data[:,j,3], marker='.', ls='-',  color=colors[j])
    ax4a.plot(data[:,j,4], speed[:,j], marker='', ls='-',  color=colors[j])
    ax4b.plot(data[:,j,4], dista[:,j], marker='', ls='-',  color=colors[j])

# Plot average speed and distance
ax4a.plot(data[:,j,4], np.average(speed[:,:], axis=1), marker='', ls='-',
          lw=2, color='k', label='average speed')
ax4b.plot(data[:,j,4], np.average(dista[:,:], axis=1), marker='', ls='-',
          lw=2, color='k', label='average distance')
ax4a.legend(loc=0)
ax4b.legend(loc=0)

# Labels etc
ax1.set_title('X-Y')
ax2a.set_title('X vs time')
ax2b.set_title('Y vs time')
ax3a.set_title('Vx vs time')
ax3b.set_title('Vy vs time')
ax4a.set_title('Speed vs time')
ax4b.set_title('Separation vs time')
ax1.set_aspect('equal')
ax4a.semilogy()
ax4b.semilogy()

fig.savefig('test1.png', bbox_inches='tight')

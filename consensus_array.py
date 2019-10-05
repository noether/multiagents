import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

c = 1

# Global options
np.random.seed(10)
do_debug = False
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Make plots from each agent reference frame
change_reference = True

# Model parameters
num_agents = 4
dim        = 3
num_steps  = 100
step_size  = 0.1

# Parameters for consensus
consensus_kc = 1e-1
consensus_kv = 1

# Generate arrays and data
pos = np.zeros((num_agents, num_steps, dim+1))   # agent, step, (t, x, y, ...)


# Random initial positions and velocities:
a = -1
b = +1
pos[:,0,1:dim+1] = (b - a) * np.random.random((num_agents, dim)) + a
# Generate initial velocity normalized to 1
v00 = np.random.random((num_agents, dim))
v0 = v00/np.linalg.norm(v00)*c
# Update first time step based on initial velocity
pos[:,1,1:dim+1] = pos[:,0,1:dim+1] + v0*step_size
pos[:,1,0] = c*step_size


# Some functions

def check_init_vel(pos, v0):
    delta_x = (pos[:,1,1:dim+1] - pos[:,0,1:dim+1])
    delta_t = (pos[:,1,0] - pos[:,0,0])[:,np.newaxis]
    np.testing.assert_almost_equal(v0, delta_x/delta_t)


def make_plot(pos, vel, outfile='test1.png'):
    times = pos[0,:,0]  # fixed time frame
    # Compute distance from converging point and speed arrays
    final_pos = np.average(pos[:,-1,1:dim+1], axis=(0))
    dista = np.linalg.norm(pos[:,:,1:dim+1] - final_pos, axis=(2)) # size = num_agents x num_steps
    speed = np.linalg.norm(vel[:,:,:], axis=(2)) # size = num_agents x num_steps

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
        ax1.plot(pos[j,:,1], pos[j,:,2], marker='.', ls='-',  color=colors[j])
        ax2a.plot(times, pos[j,:,1], marker='.', ls='-',  color=colors[j])
        ax2b.plot(times, pos[j,:,2], marker='.', ls='-',  color=colors[j])
        ax3a.plot(times[:-1], vel[j,:,0], marker='.', ls='-',  color=colors[j])
        ax3b.plot(times[:-1], vel[j,:,1], marker='.', ls='-',  color=colors[j])
        ax4a.plot(times[:-1], speed[j], marker='', ls='-',  color=colors[j])
        ax4b.plot(times[:], dista[j], marker='', ls='-',  color=colors[j])

    # Plot average speed and distance
    ax4a.plot(times[:-1], np.average(speed[:,:], axis=0), marker='', ls='-',
              lw=3, color='k', label='average speed')
    ax4b.plot(times[:], np.average(dista[:,:], axis=0), marker='', ls='-',
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
    ax1.set_xlim(-2,2)
    ax1.set_ylim(-2,2)

    fig.savefig(outfile, bbox_inches='tight')


def calc_acc(j, step, pos, consensus_kc, consensus_kv):
    acc0 = np.sum(pos[j, step-1:step, 1:dim+1] - pos[:, step-1:step, 1:dim+1], axis=0)
    vel = np.diff(pos[:,step-2:step,1:dim+1], axis=1)/np.diff(pos[:,step-2:step,0:1], axis=1)
    acc = -consensus_kc*acc0 - consensus_kv*vel[j]
    new_vel = vel[j] + acc*step_size
    delta_pos = new_vel*step_size
    return delta_pos


def lorentz_transformation(pos):
    # Following page 6 from
    # http://www.physics.umanitoba.ca/~tosborn/EM_7590_Web_Page/Resource%20Materials/Lorentz%20transformation.pdf
    vel = np.diff(pos[:,:,1:dim+1], axis=1)/np.diff(pos[:,:,0:1], axis=1)
    I = np.eye(dim)[np.newaxis, np.newaxis, :]
    beta  = vel[:,:,:,np.newaxis]/c
    betaT = vel[:,:,np.newaxis,:]/c
    beta2 = np.sum(beta**2, axis=(2), keepdims=True)
    gamma = 1./np.sqrt(1-(beta2)**2)
    trans_space = I+(gamma-1)*beta*betaT/beta2

    lorentz = np.zeros((vel.shape[0], vel.shape[1], dim+1, dim+1))
    lorentz[:,:,0,0] = gamma[:,:,0,0]
    lorentz[:,:,0,1:] = -(gamma*betaT)[0,0,0,:]
    lorentz[:,:,1:,0] = -(gamma*beta)[0,0,:,0]
    lorentz[:,:,1:,1:] = trans_space
    return lorentz


def calc_kinematics(pos):
    # From (t,x,y,z) per agent and array compute:
    # vel : 3-velocity  agent, step, (vx,vy, ...)
    # v   : speed       agent, step, ||v||
    # g   : gamma       agent, step, gamma
    # u   : 4-velocity  agent, step, (u0, ux, uy, ...)
    # tau : proper time agent, step, delta_t^2 - delta_x^2
    vel = np.diff(pos[:,:,1:dim+1], axis=1)/np.diff(pos[:,:,0:1], axis=1)
    v = np.linalg.norm(vel, axis=2)
    gamma = 1.0/np.sqrt(1-(v/c)**2)
    u   = np.zeros((pos.shape[0], pos.shape[1]-1, pos.shape[2])) # agent, step, (u0, ux, uy, ...)
    u[:,:, 1:dim+1] = gamma[:,:,np.newaxis]*vel
    u[:,:, 0] = gamma
    delta_tau = np.diff(pos[:,:,0])**2 - (np.diff(pos[:,:,1:dim+1], axis=1)**2).sum(axis=2)
    lorentz = lorentz_transformation(pos)
    return vel, v, gamma, u, delta_tau, lorentz


# Compute things:


# Generate evolution for each time step (s) for each agent (j)
for step in range(2, num_steps):
    for j in range(num_agents):
        delta_pos = calc_acc(j, step, pos, consensus_kc, consensus_kv)
        pos[j, step, 1:dim+1] = pos[j, step-1, 1:dim+1] + delta_pos
        pos[j, step, 0] = pos[j, step-1, 0] + c*step_size

# Compute kinematics from positions
vel, v, gamma, u, delta_tau, lorentz = calc_kinematics(pos)



make_plot(pos, vel, outfile = 'plot_ref.png')

# Transform all positions to reference frame of agent ref_agent:
#def galileo_transformation(pos, vel, ref_agent):
#    #speed_0 = np.linalg.norm(v[ref_agent,:,1:dim+1], axis=(1))
#    pos_tilde = np.zeros_like(pos)
#    pos_tilde[:,:,1:dim+1] = pos[:,:,1:dim+1] - pos[ref_agent,:,1:dim+1]
#    pos_tilde[:,:,0] = pos[:,:,0]
#    vel_tilde = np.zeros_like(vel)
#    vel_tilde[:,:,1:dim+1] = vel[:,:,1:dim+1] - vel[ref_agent,:,1:dim+1]
#    vel_tilde[:,:,0] = vel[:,:,0]
#    return pos_tilde, vel_tilde
#if change_reference:
#    for j in range(num_agents):
#        pos_tilde, vel_tilde = galileo_transformation(pos, vel, ref_agent=j)
#        make_plot(pos_tilde, vel_tilde, outfile = 'plot_ref_{0}.png'.format(j))


#def lorentz_transformation(vel_i):
#    I = np.eye(dim)
#    beta  = vel_i[:,np.newaxis]/c
#    betaT = vel_i[np.newaxis,:]/c
#    beta2 = np.sum(beta**2, keepdims=True)
#    gamma = 1./np.sqrt(1-(beta2)**2)
#    trans_space = I+(gamma-1)*beta*betaT/beta2
#
#    lorentz = np.zeros((dim+1, dim+1))
#    lorentz[0,0] = gamma[0,0]
#    lorentz[0,1:] = -(gamma*betaT)[0,:]
#    lorentz[1:,0] = -(gamma*beta)[:,0]
#    lorentz[1:,1:] = trans_space
#    return lorentz
#np.array([np.dot(lorentz_transformation(vel[0,step]), pos[0,step]) for step in
#          range(0,num_steps-1)])

"""
A = (-1 0)
    ( 0 1)

v1 = (1
      0)
v2 = (0
      1)
v3 = (1
      1)

v1' = (-1
        0)
v2' = (0
       1)
v3' = (-1
        1)

"""


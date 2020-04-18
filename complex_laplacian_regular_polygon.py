import numpy as np
from scipy.integrate import odeint
from scipy import linalg as la

import matplotlib.pyplot as plt
import matplotlib

np.core.arrayprint._line_width = 150


# Happy pdf for a happy submission without complains
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

# Wrapper for the integration of complex ODE
def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z

def buildLaplacian(n,ps,B):
    L = np.zeros((n,n), dtype=np.complex_)

    theta = np.pi - np.pi*(n-2)/n
    wij = np.exp(np.complex(0,-theta/2))
    wik = np.exp(np.complex(0,theta/2))

    for i in range(n-1):
        L[i,i-1] = -wij
        L[i,i+1] = -wik
        L[i,i] = wij+wik

    L[n-1,n-2] = -wij
    L[n-1,0] = -wik
    L[n-1,n-1] = wij+wik

    return L

def buildB(n):
    B = np.eye(n, dtype=np.complex_)
    for i in range(n-1):
        B[i+1,i] = -1+0j
    B[0,n-1] = -1+0j
    return B

def buildMr(n):
    Mr = np.eye(n, dtype=np.complex_)
    for i in range(n-1):
        Mr[i+1,i] = 1+0j
    Mr[0,n-1] = 1+0j
    return Mr

def buildMs(n):
    Ms = np.eye(n, dtype=np.complex_)
    for i in range(n-1):
        Ms[i+1,i] = -1+0j
    Ms[0,n-1] = -1+0j
    return Ms

def buildMt_vparallel12_sq():
    Mt = np.zeros((n,n), dtype=np.complex_)
    Mt[0,0] = 1+0j #mu 12
    Mt[1,0] = 1+0j #mu 21
    Mt[2,2] = -1+0j #mu 34
    Mt[3,2] = -1+0j #mu 43
    return Mt

def buildM_enclosing_sq():
    Menc = np.zeros((n,n), dtype=np.complex_)
    Menc[0,0] = 1+0j #mu 12
    Menc[1,0] = np.exp(np.complex(0,-np.pi/4)) #mu 21
    Menc[2,1] = 1+0j #mu 32
    return Menc


def plot_edges(fig, X, B):
    agents, edges = B.shape
    f = plt.figure(fig)
    a, b = 0, 0

    for i in range(0, edges):
        for j in range(0, agents):
            if B[j,i] == 1:
                a = j
            elif B[j,i] == -1:
                b = j

        plt.plot([X[a].real, X[b].real], [X[a].imag, X[b].imag], 'k--', lw=1.5)

def build_pstar_polygon(n):
    ps = np.zeros(n,dtype=np.complex_)
    theta = np.pi - np.pi*(n-2)/n
    for i in range(1,n):
        ps[i] = ps[i-1] + np.exp(np.complex(0,-theta*i))

    return ps

def formationControl(p, t, L):
    return -L.dot(p)

def formationControl_heading(p, t, L):
    z_12_star = 1 + 0j
    if(t > 50):
        z_12_star = np.exp(np.complex(0,np.pi/4))*z_12_star
    if(t > 100):
        z_12_star = np.exp(np.complex(0,np.pi/4))*z_12_star
    if(t > 150):
        z_12_star = np.exp(np.complex(0,np.pi/4))*z_12_star
    if(t > 200):
        z_12_star = 4*np.exp(np.complex(0,np.pi/4))*z_12_star

    heading_control = np.zeros(p0.size,dtype=np.complex_)
    heading_control[0] = (p[0]-p[1]) - z_12_star
    return -L.dot(p) - heading_control

if __name__ == "__main__":

    n = 4
    ps = build_pstar_polygon(n)
    Mr = buildMr(n)
    Ms = buildMs(n)
    Mt = buildMt_vparallel12_sq()
    Menc = buildM_enclosing_sq()

    B = buildB(n)
    L = buildLaplacian(n,ps,B)

    R = 1/(2*np.sin(np.pi/n)) # Norm of p_i in the reference shape
    ia = (n-2)*np.pi/n # Internal angle

    kr = 0 * 0.025 * R / (2*np.sin(ia/2)) # Gain that determines the angular speed in shaped consensus
    ks = 0 * -0.025 * R / (2*np.cos(ia/2)) # Gain that determines the scaling speed in shaped consensus
    kt = 0.25; # Gain for vel translation
    ke = 0.025; # Gain for enclosing

    Lmod = L -kr*Mr.dot(B.T) -ks*Ms.dot(B.T)
    Lmod_vel = L -kt*Mt.dot(B.T)
    Lmod_enclosing = L -ke*Menc.dot(B.T)

    t = np.linspace(0, 300, 10001)
    p0 = np.random.uniform(-10,10,n) + 1j*np.random.uniform(-10,10,n)

    #p, infodict = odeintz(formationControl, p0, t, args=(Lmod,), full_output=True) # Shaped Consensus
    #p, infodict = odeintz(formationControl_heading, p0, t, args=(Lmod_vel,), full_output=True) # Vel heading
    p, infodict = odeintz(formationControl, p0, t, args=(Lmod_enclosing,), full_output=True) # Enclosing

    agentcolor = ['r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g', 'b', 'r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g', 'b', 'r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g', 'b']
    plt.clf()
    plt.figure(0)
    for i in range(n):
        plt.plot(ps[i].real, ps[i].imag, 'o'+agentcolor[i], mew=1)
    plot_edges(0,ps,B.real)
    plt.axis("equal")
    plt.title("Reference shape")
    plt.grid(True)

    #time_stamps = (1500,3500,5500,7500,9500)
    time_stamps = ()
    plt.figure(1)
    for i in range(n):
        plt.plot(p[:9500,i].real, p[:9500,i].imag, agentcolor[i])
        plt.plot(p[0,i].real, p[0,i].imag, 'x'+agentcolor[i], mew=1)
        plt.plot(p[9500,i].real, p[9500,i].imag, 'o'+agentcolor[i], mew=1)
        for ts in time_stamps:
            plt.plot(p[ts,i].real, p[ts,i].imag, 'o'+agentcolor[i], mew=1)

    plot_edges(1,p[9500,:],B.real)
    for ts in time_stamps:
        plot_edges(1,p[ts,:],B.real)

    #plt.text(p[1500,1].real, p[1500,1].imag-2, 't = ' + str(t[1500]))
    #plt.text(p[3500,0].real+2, p[3500,0].imag, 't = ' + str(t[3500]))
    #plt.text(p[5500,1].real+2, p[5500,1].imag, 't = ' + str(t[5500]))
    #plt.text(p[7500,i].real+2, p[7500,i].imag, 't = ' + str(t[7500]))
    #plt.text(p[9500,i].real, p[9500,i].imag-2, 't = ' + str(t[9500]))

    plt.axis("equal")
    plt.title("Agents' trajectory")
    plt.grid(True)

# If you need Zoom, e.g., for shaped consensus

#    a = plt.axes([0.1,0.05,0.20*1.5,0.30*1.5])
#    xavg = sum(p[-1,:].real)/n
#    yavg = sum(p[-1,:].imag)/n
#    a.set_xlim(xavg-0.025,xavg+0.025)
#    a.set_ylim(yavg-0.025,yavg+0.025)
#    for i in range(n):
#        a.plot(p[:,i].real, p[:,i].imag, agentcolor[i])
#        a.plot(p[0,i].real, p[0,i].imag, 'x'+agentcolor[i], mew=1)
#        a.plot(p[-1,i].real, p[-1,i].imag, 'o'+agentcolor[i], mew=1)

#    a.grid(True)
#    plot_edges(1,p[-1,:],B.real)

    plt.show()

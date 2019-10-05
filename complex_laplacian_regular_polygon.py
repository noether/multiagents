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

if __name__ == "__main__":

    n = 10
    ps = build_pstar_polygon(n)
    Mr = buildMr(n)
    Ms = buildMs(n)
    B = buildB(n)
    L = buildLaplacian(n,ps,B)

    R = 1/(2*np.sin(np.pi/n)) # Norm of p_i in the reference shape
    ia = (n-2)*np.pi/n # Internal angle

    kr = 0.025 * R / (2*np.sin(ia/2))
    ks = 0.025 * R / (2*np.cos(ia/2))

    Lmod = L -kr*Mr.dot(B.T) -ks*Ms.dot(B.T)

    t = np.linspace(0, 250, 10001)
    p0 = np.random.uniform(-10,10,n) + 1j*np.random.uniform(-10,10,n)
    p, infodict = odeintz(formationControl, p0, t, args=(Lmod,), full_output=True)

    agentcolor = ['r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g', 'b', 'r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g', 'b', 'r', 'g', 'b', 'k', 'm', 'c', 'y', 'r', 'g', 'b']
    plt.clf()
    plt.figure(0)
    for i in range(n):
        plt.plot(ps[i].real, ps[i].imag, 'o'+agentcolor[i], mew=1)
    plot_edges(0,ps,B.real)
    plt.axis("equal")
    plt.title("Reference shape")
    plt.grid(True)

    plt.figure(1)
    for i in range(n):
        plt.plot(p[:,i].real, p[:,i].imag, agentcolor[i])
        plt.plot(p[0,i].real, p[0,i].imag, 'x'+agentcolor[i], mew=1)
        plt.plot(p[-1,i].real, p[-1,i].imag, 'o'+agentcolor[i], mew=1)

    plot_edges(1,p[-1,:],B.real)
    plt.axis("equal")
    plt.title("Agents' trajectory")
    plt.grid(True)

    a = plt.axes([0.7,0.65,0.20,0.30])
    xavg = sum(p[-1,:].real)/n
    yavg = sum(p[-1,:].imag)/n
    a.set_xlim(xavg-0.25,xavg+0.25)
    a.set_ylim(yavg-0.25,yavg+0.25)
    for i in range(n):
        a.plot(p[:,i].real, p[:,i].imag, agentcolor[i])
        a.plot(p[0,i].real, p[0,i].imag, 'x'+agentcolor[i], mew=1)
        a.plot(p[-1,i].real, p[-1,i].imag, 'o'+agentcolor[i], mew=1)

    a.grid(True)
    plot_edges(1,p[-1,:],B.real)

    plt.show()

# My own huber penalty example
# --

from cvxopt import solvers, matrix
from scipy import (
    cos, sin, linspace, pi, sqrt, zeros, dot, eye, concatenate, ones_like, ones,
    array, rand)
from scipy.stats import laplace, norm
from scipy.linalg import lstsq



# Define a function
# --
G = .4 #.65                          # asymtetry

def hgfun(theta, g=G):
    "return the value of the hg funciton"
    top = 1 - g**2
    bot = (1 + g**2 - 2*g*cos(theta))
    return top / bot

# sample
Ny = 500
hgx = linspace(0, pi, Ny)
hgy = hgfun(hgx)

# doublesided
_mix = .9
_Gback = .7
hgy = _mix * hgfun(hgx, g=G) + (1-_mix) * hgfun(hgx[-1::-1], g=_Gback)




# Noise samples
LAPLACE_PROBABILITY = .2
laplace_indicator = (rand(Ny) <= LAPLACE_PROBABILITY)

# sigma for each distribution
hgsigma = .05 * sqrt(hgy)
laplace_sigma = 4*hgsigma
samples_laplace = laplace().rvs(hgy.size) * laplace_sigma

normal_sigma = hgsigma
samples_normal = norm().rvs(hgy.size) * normal_sigma

hgnoise = (
    laplace_indicator * samples_laplace + (1-laplace_indicator) * samples_normal)

# Noisy data definition
# --
hgy_noisy = hgy + hgnoise


# Forward model
# -----------------

# lets use sin and cos
# --
NUMBER_TERMS = 11
N = NUMBER_TERMS
A = zeros((N, hgx.size))
xshift_alpha = 0.2
xshift = lambda x: x + xshift_alpha * x * (x-.5*pi) * (x-pi)

# cosine series
A[:] = [cos(n * xshift(hgx)) for n in range(N)]



# Least squares fit
# --
coef_lsq = lstsq(A.T, hgy_noisy)[0]
hgy_lsq = dot(A.T, coef_lsq)

# Huber fit
# --
HUBER_M = 1e-100 * sqrt((hgsigma**2).mean()) #0.01
# HUBER_M = 
M = 2*hgy.size + N

# subvector mappings
DU =  eye(N, M)
DT = eye(hgy.size, M, k=N)
DS = eye(hgy.size, M, k=N+Ny)

_A = 0*DT
_A[:, :N] = A.T

G = zeros((5*hgy.size, M))
G[:,:] = concatenate([
        +DT, 
        -DT, 
        _A - DT - DS,
        -_A + DT - DS, 
        -DS])

h = zeros((5*hgy.size, 1))
h[0:2*hgy.size] = HUBER_M
h[2*hgy.size:3*hgy.size, 0] = hgy_noisy
h[3*hgy.size:4*hgy.size, 0] = -hgy_noisy


# calling cvxopt
# --
GG = matrix(G, (5*hgy.size, M))
hh = matrix(h, (5*hgy.size, 1))
PP = matrix(2 * dot(DT.T, DT), (M,M))
qq = matrix(DS.sum(0), (M,1))
sol = solvers.qp(PP, qq, GG, hh)

# CVXOPT solution
u_huber = dot(DU, array(sol['x']).reshape(-1))
hgy_huber = dot(A.T, u_huber)




# 
if __name__=="__main__":
    
    # import matplotlib
    from matplotlib.pyplot import figure


    # plot the least squares vs. true
    f0 = figure(0, (8, 8), facecolor='white')
    f0.clf()
    axbox = [.05, .05, .9, .9]
    ax = f0.add_axes(axbox)
    ax.plot(hgx, hgy, 'k', linewidth=3)
    ax.plot(hgx, hgy_noisy, linewidth=0, marker='.')
    ax.plot(hgx, hgy_lsq, 'b',linewidth=1)
    ax.plot(hgx, hgy_huber, 'r', linewidth=2)
    f0.show()
    

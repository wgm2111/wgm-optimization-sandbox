
# 
# William GK Martin
# wgm2111 at columbia dot edu
# Copyright 2015
# Lisence: GPL 2
#


"""
Example fitting a noisy, scattering function with a cosine series.  
The regularization term uses a Huber penalty function, but not explicitly. 
The problem is transformed as in the textbook exercise 4.5 
  [Boyd & Vandenberghe](http://www.stanford.edu/~boyd/cvxbook)


"""

# Imports
from cvxopt import solvers, matrix
from scipy import (
    pi,
    arange, linspace, zeros, ones_like, ones,
    array, rand,
    eye, concatenate, dot,
    exp, cos, sin, sqrt)

from scipy.stats import laplace, norm
from scipy.linalg import lstsq






# Parameters to adjust
# ===========================================================================

# Function to fit
# ----
G = .55                          # henyey greenstien forward asymetry 
_mix = .95                       # parameter for mixing in some backward peak
_Gback = .7                      # back-peak asymetry factor

# Function sampling for the fit
# ----
Ny = 1000                       # number of complete samples
SUBSAMPLE_PROBABILITY = .05     # fraction of points to use in fit

# Noise parameters
# ----
SHOT_NOISE = 0.01
LAPLACE_PROBABILITY = 0.0        # Prob. of adding Laplacian to Normal noise
LAPLACIAN_SIGMA_SCALE = 1.0      # scale the stdev of Laplacian up or down

# Cosine fitting parameters
# ----
NUMBER_TERMS = 150

# Option to skew cosine towards forward/back scattering
# ----
xshift_alpha = 0.0              # 0.0 uses the standard cosine function
xshift = lambda x: x + xshift_alpha * x * (x-.5*pi) * (x-pi)

# Fitting parameters
# ----
HUBER_M = .2
L1_REGULARIZE_COEF =  NUMBER_TERMS / (3 * float(Ny))
L2_RESIDUAL_COEF = 1.0 



# Script
# ==========================================================================

# Define the Henyey Greenstien function to fit 
def hgfun(theta, g=G):
    "return the value of the hg funciton"
    top = 1 - g**2
    bot = (1 + g**2 - 2*g*cos(theta))
    return top / bot

# Sample the function 
hgx = linspace(0, pi, Ny)
hgy = hgfun(hgx)

# Use a double sided HG function 
hgy = _mix * hgfun(hgx, g=G) + (1-_mix) * hgfun(hgx[-1::-1], g=_Gback)

# Take a subsample of the function evaluations to use in the fit
subsample_indicator = (rand(Ny) <= SUBSAMPLE_PROBABILITY)
_hgx = hgx
_hgy = hgy
hgx = hgx[subsample_indicator]
hgy = hgy[subsample_indicator]
Ny = hgy.size



# Noise samples (Mix of Laplacian and Normal)
# ------------------------------
laplace_indicator = (rand(Ny) <= LAPLACE_PROBABILITY)
hgsigma = SHOT_NOISE * sqrt(hgy) 
laplace_sigma = LAPLACIAN_SIGMA_SCALE * hgsigma
samples_laplace = laplace().rvs(hgy.size) * laplace_sigma
normal_sigma = hgsigma
samples_normal = norm().rvs(hgy.size) * normal_sigma
hgnoise = (
    laplace_indicator * samples_laplace + (1-laplace_indicator) * samples_normal)
hgy_noisy = hgy + hgnoise


# Forward model
# -----------------

# lets use sin and cos
# --
N = NUMBER_TERMS
A = zeros((N, hgx.size))
__A = zeros((N, _hgx.size))

# Basis evaluation arrays
A[:] = [cos(n * xshift(hgx)) for n in range(N)]
__A[:] = [cos(n * xshift(_hgx)) for n in range(N)] # at randomly sampled points



# Fitting 
# ==================================================================

# Least squares fit
# --

coef_lsq = lstsq(A.T, hgy_noisy)[0]
hgy_lsq = dot(__A.T, coef_lsq)


# Fitting with Huber penalty and L1 prior
M = 2*Ny + 2*N

# subvector mappings
DU =  eye(N, M)
DT = eye(Ny, M, k=N)
DS = eye(Ny, M, k=N+Ny)
DZ = eye(N, M, k=N+Ny+Ny)

# make an _A array
# --
_A = 0*DT
_A[:, :N] = A.T


# Construct the G array
# --
Gdim0 = 5*Ny + 3*N
Gdim1 = M
Gshape = (Gdim0, Gdim1)

# instantiate and fill
G = zeros(Gshape)
G[:,:] = concatenate([
        (DT), 
        -DT, 
        (_A - DT - DS),
        -_A + DT - DS, 
        -DS, 
        (DU - DZ), 
        -DU - DZ, 
        -DZ])

# The right hand side for G
# --

def get_huber_fit(L2_RESIDUAL_COEF, L1_REGULARIZE_COEF):
    "Run the code with given parameters "

    h = zeros((Gdim0, 1))
    h[0:2*Ny] = HUBER_M
    h[2*Ny:3*Ny, 0] = hgy_noisy
    h[3*Ny:4*Ny, 0] = -hgy_noisy
    
    
    # calling cvxopt
    # --
    GG = matrix(G, Gshape)
    hh = matrix(h, (Gdim0, 1))
    PP = matrix(2 * dot(DT.T, DT) * L2_RESIDUAL_COEF, (M,M))
    qq = matrix(DS.sum(0) + L1_REGULARIZE_COEF * DZ.sum(0), (M,1))
    sol = solvers.qp(PP, qq, GG, hh)
    
    # CVXOPT solution
    u_huber = dot(DU, array(sol['x']).reshape(-1))
    hgy_huber = dot(__A.T, u_huber)
    fit_hgy_huber = dot(A.T, u_huber)

    return u_huber, hgy_huber, fit_hgy_huber, sol

u_huber, hgy_huber, fit_hgy_huber, sol = get_huber_fit(L2_RESIDUAL_COEF, L1_REGULARIZE_COEF)



# 
if __name__=="__main__":
    
    # import matplotlib
    from matplotlib.pyplot import figure


    # plot the least squares vs. true
    f0 = figure(0, (8, 15), facecolor='white')
    f0.clf()
    axbox = [.05, .05, .9, .9]
    ax = f0.add_axes(axbox)
    ax.plot(_hgx, _hgy, 'k', linewidth=5)
    ax.plot(hgx, hgy_noisy, linewidth=0, marker='.',markersize=5)
    ax.plot(_hgx, hgy_lsq, 'b',linewidth=1)
    ax.plot(_hgx, hgy_huber, 'r', linewidth=2)
    ax.plot(hgx, fit_hgy_huber, 'r', linewidth=0, marker='o')#, markersize=12)
    ax.set_ybound(-.3, hgy.max()*7.0/6.0)
    ax.set_xbound(0, pi)
    f0.show()


    # Make a tradeoff curve
    # --
    trade_offs = 15
    L1_coefs = 2**(linspace(-12, 2,trade_offs)) * sol['primal objective']#sqrt(((fit_hgy_huber-hgy_noisy)**2).sum())
    l1norms = zeros(trade_offs)
    resnorms = zeros(trade_offs)
    errnorms = zeros(trade_offs)
    hgy_tradeoffs = zeros((trade_offs,_hgx.size))


    for i, L1_coef in zip(range(trade_offs), L1_coefs):
        u, hgy_fit_fine, hgy_fit_coarse, sol = get_huber_fit(L2_RESIDUAL_COEF, L1_coef)
        l1norms[i] = abs(u).sum()
        resnorms[i] = sqrt(((hgy_fit_coarse-hgy_noisy)**2).sum())
        errnorms[i] = sqrt(((hgy_fit_coarse-hgy)**2).sum())
        hgy_tradeoffs[i] = hgy_fit_fine
        
        
    sumnorms = 2 * l1norms + resnorms
    I = arange(trade_offs)[min(sumnorms)==sumnorms][0]
    ax.plot(_hgx, hgy_tradeoffs[I], '-g', linewidth=5)
        

    f1 = figure(1, (13, 13), facecolor='white')
    f1.clf()
    ax2 = f1.add_axes(axbox)
    ax2.plot(l1norms, resnorms, 'k', linewidth=5, marker='o')
    ax2.plot(l1norms, errnorms, 'r', linewidth=2, marker='o')
    f1.show()

    

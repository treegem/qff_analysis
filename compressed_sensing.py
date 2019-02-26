"""
This file contains functions to reconstruct data by compressed sensing
"""

import numpy as np
from scipy.constants import mu_0, pi
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
#import cvxpy as cvx



#%% apply compressed sensing to fake signal 
if __name__ == '__main__':

    #%% create model of spins
    rs = np.linspace(0,5,101) #coordinates relative to wire
    
    spins_true = np.zeros_like(rs)  #fake distribution of spins
    r0 = 3.0
    sr = .4
    spins_true += np.exp(-(rs - r0)**2/2/sr**2) #broad peak from background spins
    spins_true[30] += 1.0 #single spin
    spins_true[40] += 1.0
    
    #simulate NV signal
    y = spfft.idct(spins_true, norm='ortho') #NV signal
    fx = spfft.fftfreq(len(spins_true), d=spins_true[1]-spins_true[0])
    
    #add readout noise
    y += 0.2*(np.random.rand(len(spins_true)) -.5)
    
    #%% estimate spin density by IDCT

    plt.figure()
    plt.plot(spfft.fftshift(fx), spfft.fftshift(y))
    plt.xlabel("fx (1/um)")
    plt.ylabel("NV signal")
    plt.show()
    
    plt.figure()
    plt.plot(rs, spfft.dct(y, norm='ortho'))
    plt.plot(rs, spins_true)
    plt.xlabel("r (2pi/um)")
    plt.ylabel("spin density")
    plt.show()
    
    #%% create thinned spin density to test solvers

    m = int(.4*len(rs)) # length of sparse sample
    n=len(rs)
    ris = np.random.choice(n, m, replace=False) # random sample of indices
    ris = np.unique(np.hstack((np.arange(6), ris))) #add first 10 samples and sort
                                                    #(to give a stronger prior on the frequency range)
    
    fx_thinned = fx[ris]
    y_thinned = y[ris]
    
    # create idct matrix operator
    A = spfft.idct(np.identity(n), norm='ortho', axis=0)
    A = A[ris]
    
    #plt.figure()
    #plt.plot(np.dot(A, spins_true))
    #plt.plot(y_thinned)
    #plt.plot("frequency index")
    #plt.plot("NV signal")
    #plt.show()
    
    #%% estimate spin density by Lasso (L1-regularized L2 minimisation)
    lasso_solver = Lasso(alpha=.002, tol=1e-5) #alpha = .001
    lasso_solver.fit(A, y_thinned)
    
    plt.figure()
    plt.subplot(121)
    plt.plot(lasso_solver.coef_)
    plt.plot(-spins_true)
    plt.subplot(122)
    plt.plot(np.dot(A, lasso_solver.coef_))
    plt.plot(y_thinned)
    plt.show()
    
    #%% estimate spin density by orthogonal matching pursuit
    omp_solver = OrthogonalMatchingPursuit(tol=1e-3)
    omp_solver.fit(A, y_thinned)
    plt.figure()
    plt.subplot(121)
    plt.plot(omp_solver.coef_)
    plt.plot(-spins_true)
    plt.subplot(122)
    plt.plot(np.dot(A, omp_solver.coef_))
    plt.plot(y_thinned)
    plt.show()
    
    #%% estimate spin density by L1 minimization 
    # do L1 optimization
    #vx = cvx.Variable(n)
    #objective = cvx.Minimize(cvx.norm(vx, 1))
    #constraints = [A*vx == y2]
    #prob = cvx.Problem(objective, constraints)
    #spins_l1 = prob.solve(verbose=True)
    
    
    #%% estimate spin density by Bayesian analysis - does not work so far
    
    p_r  = np.ones_like(rs)/n #estimated spin density in real space
    
    p_fx = .5* (1 + y/y[0])  #evidence: p(Sz=0)@fx
    
    p_fx_bar_r = spfft.idct(np.identity(n))
    p_fx_bar_r = .5*(1 + p_fx_bar_r/p_fx_bar_r.max())
    
    #plt.figure()
    #plt.plot(fx, p_fx)
    #plt.show()
    
    plt.figure() 
    plt.imshow(p_fx_bar_r)
    plt.colorbar()
    plt.plot()
    
    for fxi, p_fx_point in enumerate(p_fx):
        p_r *= p_fx_bar_r[:,fxi]/p_fx_point
    
    plt.figure()
    plt.plot(p_fx_bar_r[50,:])
    plt.show()
    
    #plt.figure()
    #plt.plot(p_r[1:])
    #plt.show()
    
    #%% check DCT normalization
    plt.figure()
    plt.plot(rs, spfft.dct(spfft.idct(spins_true, norm='ortho'), norm='ortho'))
    

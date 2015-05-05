#!/usr/bin/env ipython2

import numpy as np
import scipy.special as ss
import scipy.interpolate as sint
from statepoint import StatePoint
from matplotlib import pyplot as plt
from uncertainties import ufloat
from gen_mgxs import mgxs
import pickle
from bisect import bisect
import os
import sys

def Pn_solve(sigtn, sigsn, Qn, deriv_term):
    # d/dx[(n/(2n+1))*psinm1+(n+1)/(2n+1)*psip1]+sigtn*psin=sigsn*psin+Qn
    # deriv_term+sigtn*psin=sigsn*psin+Qn
    # psin = (Qn-deriv_term)/(sigtn-sigsn)

    # psin = (Qn - deriv_term) / (sigtn - np.sum(sigsn,axis=0))
    psin = (Qn - deriv_term) / (sigtn)

    return psin[:]

def solve1g(N, sigtns1g, sigsns1g, Qnsg, psinsg, x, invdx, n_ratios):
    # Loops through each of the n orders, sets up the derivitave term,
    # and calls Pn_solve on it.

    # n_ratios is [(n/(2n+1),(n+1)/(2n+1)) for n in range(N+1)]

    for n in xrange(N+1):
        # N+1 so we get an n==N in this loop

        # Set up the deriv_term.
        # Due to assumed reflective BC, deriv_term will always be 0 for
        # ix ==0 and ix == last one, so we can skip those
        if n > 0:
            nm1_interp = sint.KroghInterpolator(x, psinsg[n - 1])
        else:
            nm1_interp = sint.KroghInterpolator([x[0],x[-1]],[0.0, 0.0])
        if n < N:
            np1_interp = sint.KroghInterpolator(x, psinsg[n + 1])
        else:
            np1_interp = sint.KroghInterpolator([x[0],x[-1]],[0.0, 0.0])
        deriv_term = n_ratios[n][0] * nm1_interp.derivative(x) + \
                     n_ratios[n][1] * np1_interp.derivative(x)

        # Now adjust for BC
        deriv_term[0] = 0.0
        deriv_term[-1] = 0.0

        # Now we can pass this to Pn_solve to get our new psin values
        psinsg[n,:] = Pn_solve(sigtns1g[n], sigsns1g[n], Qnsg[n], deriv_term)

    return psinsg[:,:]

def fixedsrc(N, G, sigtns, sigsns, Qns, psins, x, invdx, n_ratios, eps_psi, max_inner):
    # Not implemented yet. This wll be the MG solver.
    eps = 1.0E4
    iter = 0
    newQns = np.zeros_like(Qns[:,0,:])
    # import pdb; pdb.set_trace()
    while ((eps > eps_psi) and (iter <= max_inner)):
        # Develop scattering source
        for g in range(G):
            for n in range(N):
                for ix in range(len(invdx) + 1):
                    newQns[n,ix] = Qns[g,n,ix] + \
                        np.dot(sigsns[:,n,g,ix], psins[:,n,ix])
            # Run fixed src solver
            psins[g,:,:] = solve1g(N, sigtns[g,:,:], sigsns[g,:,:,:], newQns,
                                   psins[g,:,:], x, invdx, n_ratios)

        # eps =
        iter += 1
        for g in xrange(G):
            plt.plot(x,psins[g,0,:],label='Pn')
            plt.plot(x,omcflux[g,0,:],label='OMC')
            plt.legend(loc='best')
            plt.show()
            plt.close()
    print "Inner Iterations = " + str(iter)

def init(x, G, N, flux_guess):
    invdx = np.zeros(len(x) - 1)
    for ix in xrange(len(invdx)):
        invdx[ix] = 1.0 / (x[ix + 1] - x[ix])

    n_ratios = [(float(n)/float(2 * n + 1), float(n + 1)/float(2 * n + 1))
                for n in range(N + 1)]

    psins = np.ones(shape=(G, N + 1, len(x)))
    for g in xrange(G):
        for n in xrange(N + 1):
            psins[g,n,:] = flux_guess[g,n,:] / np.sum(flux_guess[g,n,:])

    return invdx, n_ratios, psins

def get_openmc_mesh(spFile, tid, sid, G, N, extent):
    sp = StatePoint(spFile)
    sp.read_results()
    sp.generate_stdev()

    keff = ufloat(sp.k_combined[0], sp.k_combined[1])

    GN = [[0.0 for n in xrange(N)] for g in xrange(G)]
    data = np.array(GN[:][:])
    dx = extent / float(N)
    x = [(float(i) + 0.5) * dx for i in xrange(N)]
    for g in xrange(G):
        myg = G - g - 1
        for n in xrange(N):
            m, u = sp.get_value(tid, [('mesh',(1,1,n+1)),('energyin',g)], sid)
            data[myg,n] = m

    return x, data[:,:], keff

def get_openmc_mesh_matrix(spFile, tid, sid, G, N, extent):
    sp = StatePoint(spFile)
    sp.read_results()
    sp.generate_stdev()

    keff = ufloat(sp.k_combined[0], sp.k_combined[1])

    GGN = [[[0.0 for n in xrange(N)] for go in xrange(G)] for g in xrange(G)]
    data = np.array(GGN[:][:][:])
    dx = extent / float(N)
    x = [(float(i) + 0.5) * dx for i in xrange(N)]
    for g in xrange(G):
        myg = G - g - 1
        for go in xrange(G):
            mygo = G - go - 1
            for n in xrange(N):
                m, u = sp.get_value(tid, [('mesh',(1,1,n+1)),('energyin',g),
                                    ('energyout',go)], sid)
                data[myg,mygo,n] = m

    return x, data[:,:,:], keff

def get_omc_mgxs(sp, mesh_tids, mesh_sids, order, G, Nmesh, extent, xstype):
    # Get flux-yN
    fluxyn = np.zeros(shape=(order, G, Nmesh))
    for l in range(order):
        tid = mesh_tids[0]
        sid = mesh_sids[0][l]
        x, fluxyn[l,:,:], omck = get_openmc_mesh(sp,tid,sid,G,Nmesh,extent)

    # Get scatt-pN
    scattpn = np.zeros(shape=(order, G, G, Nmesh))
    for l in range(order):
        tid = mesh_tids[1]
        sid = mesh_sids[1][l]
        x, scattpn[l,:, :, :], omck = get_openmc_mesh_matrix(sp,tid,sid,G,Nmesh,extent)

    # Get scatt-yN
    scattyn = np.zeros(shape=(order, G, G, Nmesh))
    for l in range(order):
        tid = mesh_tids[2]
        sid = mesh_sids[2][l]
        x, scattyn[l,:,:,:], omck = get_openmc_mesh_matrix(sp,tid,sid,G,Nmesh,extent)

    # Get total-yN
    totalyn = np.zeros(shape=(order, G, Nmesh))
    for l in range(order):
        tid = mesh_tids[3]
        sid = mesh_sids[3][l]
        x, totalyn[l,:,:], omck = get_openmc_mesh(sp,tid,sid,G,Nmesh,extent)

    # Get nu-fission (right now only doing iso weighting)
    nusigfns = np.zeros(shape=(order, G, G, Nmesh))
    tid = mesh_tids[4]
    sid = mesh_sids[4][0]
    # Now only doing iso weighting so l=0
    x, nusigfns[0,:,:,:], omck = get_openmc_mesh_matrix(sp,tid,sid,G,Nmesh,extent)

    Qns = np.zeros(shape=(order, G, Nmesh))
    # put Q in nusigfns, leave as isotropic now
    l = 0
    Qsum = 0.0
    for go in range(G):
        for n in range(Nmesh):
            Qns[l,go,n] = 0.0
            for g in range(G):
                Qns[l,go,n] += nusigfns[0,g,go,n]
            Qsum += Qns[l,go,n]
    Qns[l,:,:] /= Qsum
    for l in range(1,order):
        for g in range(G):
            for n in range(Nmesh):
                Qns[l,g,n] = 0.0

    totaliso = totalyn[0,:,:]
    for l in range(order):
        for g in range(G):
            for n in range(Nmesh):
                # Nmeshormalize by flux
                flux = fluxyn[l,g,n]
                flux0 = fluxyn[0,g,n]
                if flux0 != 0.0:
                    for go in range(G):
                        scattpn[l,g,go,n] /= flux0
                    if l == 0:
                        totaliso[g,n] /= flux0
                if flux != 0.0:
                    for go in range(G):
                        scattyn[l,g,go,n] /= flux
                        nusigfns[l,g,go,n] /= flux
                    totalyn[l,g,n] /= flux

                # Apply correction
                if xstype == 'consP':
                    corr = totaliso[g,n] - totalyn[l,g,n]
                    for go in range(G):
                        scattyn[l,g,go,n] += corr

    if xstype == 'iso':
        sigtns = [totaliso for l in range(order)]
        sigsns = scattpn[:]
    elif xstype == 'consP':
        sigtns = [totaliso for l in range(order)]
        sigsns = scattyn[:]
    elif xstype == 'yN':
        sigtns = totalyn[:]
        sigsns = scattyn[:]

    return omck, np.swapaxes(fluxyn,0,1), x, np.swapaxes(sigtns,0,1), \
        np.swapaxes(sigsns,0,1), np.swapaxes(nusigfns,0,1), np.swapaxes(Qns,0,1)


if __name__ == "__main__":
    rcdef = plt.rcParams.copy
    newparams = {'savefig.dpi': 100, 'figure.figsize': (24, 13.5),
                 'font.size': 16}
    plt.rcParams.update(newparams)

    if len(sys.argv) != 3:
        raise ValueError("Must Provide Cross-Section Type [consP, iso, yN] & " +
                         "Run Type [FS, k]!")
    else:
        xstype = sys.argv[1]
        if xstype not in ["consP", "iso", "yN"]:
            raise ValueError("Invalid Cross-Section Type!")
        runtype = sys.argv[2]
        if runtype not in ["FS", "k"]:
            raise ValueError("Invalid Run Type!")


    show = False
    save = True
    G = 4
    N = 1
    Nmesh = 16
    extent = 0.64
    sp = './statepoint.08000.binary'
    eps_psi = 1.0E-6
    max_inner = 2

    # First get the mgxs data and create x/s
    if xstype == 'iso':
        momWgt = False
        trcorr = None
    elif xstype == 'consP':
        momWgt = True
        trcorr = 'consP'
    elif xstype == 'yN':
        momWgt = True
        trcorr = None

    mesh_tids = [0, 1, 1, 0, 2]
    mesh_sids = [[0,2,6,12], [0,1,2,3], [4,6,10,16], [16,18,22,27], [0]]

    omck, omcflux, x, sigtns, sigsns, nusigfns, Qns = \
        get_omc_mgxs(sp, mesh_tids, mesh_sids, N+1, G, Nmesh, extent, xstype)
    print 'OpenMC k_eff=' + "{:12.5E}".format(omck)

    # Set up some of our data we will use during the sweep
    invdx, n_ratios, psins = init(x, G, N, omcflux)

    if runtype == 'FS':
        fixedsrc(N, G, sigtns, sigsns, Qns, psins, x, invdx, n_ratios, eps_psi, max_inner)
        # Estimate k to compare with the openMC k
        pnk = 0.0
        for g in xrange(G):
            for ix in xrange(Nmesh):
                if Qns[g,0,ix] > 0.0:
                    pnk += np.sum(nusigfns[g,0,:,ix])*psins[g,0,ix] / Qns[g,0,ix]
    else:
        print "k-eigenvalue solver not yet implemented!"

    pcm = 1.0E5*(pnk-omck)/omck

    print "Pn k_eff = " + "{:12.5E}".format(pnk)

    print "pcm = " + "{:12.5E}".format(pcm)



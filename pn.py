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

    psin = (Qn - deriv_term) / (sigtn - sigs)

    return psin

def solve1g(N, sigtns1g, sigsns1g, Qnsg, psinsg, invdx, n_ratios):
    # Loops through each of the n orders, sets up the derivitave term,
    # and calls Pn_solve on it.

    # n_ratios is [(n/(2n+1),(n+1)/(2n+1)) for n in range(N+1)]

    deriv_term = np.zeros_like(invdx)

    for n in xrange(N+1):
        # N+1 so we get an n==N in this loop

        # Set up the deriv_term.
        # Due to assumed reflective BC, deriv_term will always be 0 for
        # ix ==0 and ix == last one, so we can skip those
        for ix in range(1, len(invdx) - 1):
            if n > 0:
                deriv_term[ix]  = n_ratios[n][0] * \
                    (psinsg[n - 1,ix + 1] - psinsg[n - 1,ix]) * dx[ix]
            elif n < N:
                deriv_term[ix] += n_ratios[n][1] * \
                    (psinsg[n + 1,ix + 1] - psinsg[n + 1,ix]) * dx[ix]

        # Now we can pass this to Pn_solve to get our new psin values
        psinsg[n,:] = Pn_solve(sigtns1g[n], sigsns1g[n], Qnsg[n], deriv_term)

def fixedsrc(N, G, sigtns, sigsns, Qnsg, psins, invdx, n_ratios):
    # Not implemented yet. This wll be the MG solver.
    pass

def init(x, N):
    invdx = np.zeros(len(x) - 1)
    for ix in xrange(len(invdx)):
        invdx[ix] = 1.0 / (x[ix + 1] - x[ix])

    n_ratios = [(float(n)/float(2 * n + 1), float(n + 1)/float(2 * n + 1))
                for n in range(N + 1)]

    return invdx, n_ratios

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
    fluxyn = [0.0 for l in range(order)]
    for l in range(order):
        tid = mesh_tids[0]
        sid = mesh_sids[0][l]
        x, fluxyn[l], omck = get_openmc_mesh(sp,tid,sid,G,Nmesh,extent)

    # Get scatt-pN
    scattpn = [0.0 for l in range(order)]
    for l in range(order):
        tid = mesh_tids[1]
        sid = mesh_sids[1][l]
        x, scattpn[l], omck = get_openmc_mesh_matrix(sp,tid,sid,G,Nmesh,extent)

    # Get scatt-yN
    scattyn = [0.0 for l in range(order)]
    for l in range(order):
        tid = mesh_tids[2]
        sid = mesh_sids[2][l]
        x, scattyn[l], omck = get_openmc_mesh_matrix(sp,tid,sid,G,Nmesh,extent)

    # Get total-yN
    totalyn = [0.0 for l in range(order)]
    for l in range(order):
        tid = mesh_tids[3]
        sid = mesh_sids[3][l]
        x, totalyn[l], omck = get_openmc_mesh(sp,tid,sid,G,Nmesh,extent)

    # Get nu-fission (right now only doing iso weighting)
    nusigfns = [None for l in range(order)]
    tid = mesh_tids[4]
    sid = mesh_sids[4][0]
    # Now only doing iso weighting so l=0
    x, nusigfns[0], omck = get_openmc_mesh_matrix(sp,tid,sid,G,Nmesh,extent)
    for l in range(1,order):
        nusigfns[l] = np.zeros_like(nusigfns[0])


    Qns = [np.asarray([[0.0 for n in xrange(Nmesh)] for go in xrange(G)]) for l in xrange(order)]
    # put Q in nusigfns, leave as isotropic now
    l = 0
    for go in range(G):
        for n in range(Nmesh):
            Qns[l][go,n] = 0.0
            for g in range(G):
                Qns[l][go,n] += nusigfns[0][g,go,n]
    for l in range(1,order):
        for g in range(G):
            for n in range(Nmesh):
                Qns[l][g,n] = 0.0

    totaliso = totalyn[0][:,:]
    for l in range(order):
        for g in range(G):
            for n in range(Nmesh):
                # Nmeshormalize by flux
                flux = fluxyn[l][g,n]
                flux0 = fluxyn[0][g,n]
                if flux0 != 0.0:
                    for go in range(G):
                        scattpn[l][g,go,n] /= flux0
                    if l == 0:
                        totaliso[g,n] /= flux0
                if flux != 0.0:
                    for go in range(G):
                        scattyn[l][g,go,n] /= flux
                        nusigfns[l][g,go,n] /= flux
                    totalyn[l][g,n] /= flux

                # Apply correction
                if xstype == 'consP':
                    corr = totaliso[g,n] - totalyn[l][g,n]
                    for go in range(G):
                        scattyn[l][g,go,n] += corr

    if xstype == 'iso':
        sigtns = [totaliso for l in range(order)]
        sigsns = scattpn[:]
    elif xstype == 'consP':
        sigtns = [totaliso for l in range(order)]
        sigsns = scattyn[:]
    elif xstype == 'yN':
        sigtns = totalyn[:]
        sigsns = scattyn[:]

    print totaliso[0]

    return omck, x, sigtns, sigsns, nusigfns, Qns


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
    N = 3
    Nmesh = 16
    extent = 0.64
    sp = './statepoint.10000.binary'

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

    omck, x, sigtns, sigsns, nusigfns, Qns = get_omc_mgxs(sp, mesh_tids,
                                                          mesh_sids, N+1, G,
                                                          Nmesh, extent, xstype)
    print 'OpenMC k_eff=' + "{:12.5E}".format(omck)

    # Set up some of our data we will use during the sweep
    invdx, n_ratios = init(x, N)



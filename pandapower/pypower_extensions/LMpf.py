# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
# modified by Cheng Tianshi

"""Solves the power flow using a full Newton's method.
"""

import sys

from numpy import (Inf, angle, arange, array, conj, empty, exp, float64, int64,
                   linalg, r_, zeros, eye)
from pypower.ppoption import ppoption
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, issparse, vstack
from scipy.sparse.linalg import spsolve


def LMpf(Ybus, Sbus, V0, ref, pv, pq, ppopt=None, numba=True):
    """Solves the power flow using a LM's method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles. C{ppopt} is a PYPOWER options vector which can be used to
    set the termination tolerance, maximum number of iterations, and
    output options (see L{ppoption} for details). Uses default options if
    this parameter is not given. Returns the final complex voltages, a
    flag which indicates whether it converged or not, and the number of
    iterations performed.

    The levenberg—marquardt method is based on the least square. It can solve 
    both the normal and ill-condition power system. It also can return the 
    least square solution when no exact solution can be found. 

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    Modified by University of Kassel (Florian Schaefer) to use numba
    Modified to the levenberg—marquardt method by Cheng Tianshi
    """

    # default arguments
    global dSbus_dV_calc, dSbus_dV_calc
    if ppopt is None:
        ppopt = ppoption()

    # options
    tol = ppopt['PF_TOL']
    max_it = ppopt['PF_MAX_IT']
    verbose = ppopt['VERBOSE']

    # initialize
    converged = 0
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    # set up indexing for updating V
    pvpq = r_[pv, pq]

    # generate lookup pvpq -> index pvpq (used in createJ)
    pvpq_lookup = zeros(pvpq.max() + 1, dtype=int)
    pvpq_lookup[pvpq] = arange(len(pvpq))

    # if numba is available import "numba enhanced" functions
    if numba:
        from pandapower.pypower_extensions.create_J import create_J, create_J2
        from pandapower.pypower_extensions.dSbus_dV import dSbus_dV_calc
        from pandapower.pypower_extensions.dSbus_dV import dSbus_dV

        # check if pvpq is the same as pq. In this case a faster version of
        # create_J can be used
        if len(pvpq) == len(pq):
            createJ = create_J2
        else:
            createJ = create_J
    else:
        # numba == False -> Import pypower standard
        from pypower.dSbus_dV import dSbus_dV

    npv = len(pv)
    npq = len(pq)
    j1 = 0
    j2 = npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses

    # evaluate F(x0)
    mis = V * conj(Ybus * V) - Sbus
    F = r_[mis[pv].real,
           mis[pq].real,
           mis[pq].imag]

    # check tolerance
    normF = linalg.norm(F, Inf)
    if verbose > 1:
        sys.stdout.write('\n it    max P & Q mismatch (p.u.)')
        sys.stdout.write('\n----  ---------------------------')
        sys.stdout.write('\n%3d        %10.3e' % (i, normF))
    if normF < tol:
        converged = 1
        if verbose > 1:
            sys.stdout.write('\nConverged!\n')

    Ybus = Ybus.tocsr()

    alpha = 100
    m = 0.001
    p0 = 0.0001
    p1 = 0.8
    p2 = 0.95
    update = True
    # do LM iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        # use numba if activated
        if update:
            if numba:
                # create Jacobian from fast calc of dS_dV
                dVm_x, dVa_x = dSbus_dV_calc(
                    Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V))

                # data in J, space preallocated is bigger than acutal Jx ->
                # will be reduced later on
                Jx = empty(len(dVm_x) * 4, dtype=float64)
                # row pointer, dimension = pvpq.shape[0] + pq.shape[0] + 1
                Jp = zeros(pvpq.shape[0] + pq.shape[0] + 1, dtype=int64)
                # indices, same with the preallocated space (see Jx)
                Jj = empty(len(dVm_x) * 4, dtype=int64)

                # fill Jx, Jj and Jp
                createJ(dVm_x, dVa_x, Ybus.indptr, Ybus.indices,
                        pvpq_lookup, pvpq, pq, Jx, Jj, Jp)

                # resize before generating the scipy sparse matrix
                Jx.resize(Jp[-1], refcheck=False)
                Jj.resize(Jp[-1], refcheck=False)

                # generate scipy sparse matrix
                dimJ = npv + npq + npq
                J = sparse((Jx, Jj, Jp), shape=(dimJ, dimJ))

            else:
                # create Jacobian with standard pypower implementation.
                dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

                # evaluate Jacobian
                J11 = dS_dVa[array([pvpq]).T, pvpq].real
                J12 = dS_dVm[array([pvpq]).T, pq].real
                J21 = dS_dVa[array([pq]).T, pvpq].imag
                J22 = dS_dVm[array([pq]).T, pq].imag

                J = vstack([
                    hstack([J11, J12]),
                    hstack([J21, J22])
                ], format="csr")

        H = J.T * J
        NF2 = linalg.norm(F, 2)
        u = alpha * NF2
        JF = J.T * F
        # compute update step
        dx = -1 * spsolve(H + u * sparse(eye(H.shape[0])), JF)
        V_0 = array(V)
        Va_0 = array(Va)
        Vm_0 = array(Vm)
        # update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]

        V = Vm * exp(1j * Va)

        # evalute F(x)
        mis = V * conj(Ybus * V) - Sbus
        F0 = array(F)
        F = r_[mis[pv].real,
               mis[pq].real,
               mis[pq].imag]
        temp = F + J * dx
        tau = (linalg.norm(F0, 2)**2 - linalg.norm(F, 2)**2) / \
            (linalg.norm(F0, 2)**2 - linalg.norm(temp, 2)**2)

        if tau > p0:
            update = True
        else:
            update = False
            V = V_0
            Va = Va_0
            Vm = Vm_0

        if tau < p1:
            alpha *= 10
        elif p1 < tau and tau < p2:
            pass
        elif tau > p2:
            alpha = max([alpha / 10.0, m])
        # check for convergence
        normF = linalg.norm(J.T * F, 2)
        if verbose > 1:
            sys.stdout.write('\n%3d        %10.3e' % (i, normF))
        if normF < tol:
            converged = 1
            if verbose:
                sys.stdout.write("\nLM's method power flow converged in "
                                 "%d iterations.\n" % i)

    if verbose:
        if not converged:
            sys.stdout.write("\nLM's method power did not converge in %d "
                             "iterations.\n" % i)

    return V, converged, i

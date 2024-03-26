import numpy as np
import scipy as sp

from src.Utilities import Nmat, Bmat, CFEM_length, Parameters


def WaveDisp(Pr: Parameters) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[list[float]]]:
    """
    Compute wave dispersion for a given set of parameters.

    Parameters:
        Pr (Parameters): The input parameters for wave dispersion calculation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[list[float]]]:
            - kz (np.ndarray): Array of wave numbers.
            - cg (np.ndarray): Array of group velocities.
            - cpE (np.ndarray): Array of effective phase velocities.
            - cgT (list[np.ndarray]): List of group velocities with tracing.
            - cpT (list[np.ndarray]): List of phase velocities with tracing.
            - kzT (list[np.ndarray]]): List of wave numbers with tracing.
    """
    # CHECKS  -----------------------------------------------------------------
    if np.any((Pr.cs < 0) | (Pr.h < 0)):
        z = np.zeros_like(Pr.w)
        return z.reshape(1, -1), z, z, [z] * Pr.nMode, [z] * Pr.nMode, [z] * Pr.nMode
    trace, nMode = Pr.trace, Pr.nMode
    if Pr.eff == 'yes' and Pr.interpOutlier == 'yes' and Pr.trace == 'no':
        trace, nMode = 'yes', 10
    # SOLID  ------------------------------------------------------------------
    # MATERIAL
    mat_type = float
    nL = len(Pr.cs)
    GPL = np.empty(nL, dtype=object)
    if Pr.fem == 'cfem':
        Pr.order[:], mat_type = 1, complex
    for i in range(nL):
        ngp = 1 if Pr.fem == 'cfem' else Pr.order[i] + 1
        points, weights = np.polynomial.legendre.leggauss(ngp)
        GPL[i] = np.vstack((points, weights))
    nenS = Pr.order + 1

    nelS = sum(Pr.nDivS)
    D = np.empty((3, nL), dtype=object)
    Lz = np.zeros((3, 2))
    Lz[0, 0] = 1
    Lz[2, 1] = 1
    Ly = np.zeros((3, 2))
    Ly[1, 1] = 1
    Ly[2, 0] = 1
    for i in range(nL):
        Dmat = np.zeros((3, 3))
        Dmat[0, 0] = Pr.roS[i] * Pr.cp[i] ** 2
        Dmat[1, 1] = Dmat[0, 0]
        Dmat[2, 2] = Pr.roS[i] * Pr.cs[i] ** 2
        Dmat[0, 1] = Pr.roS[i] * (Pr.cp[i] ** 2 - 2 * Pr.cs[i] ** 2)
        Dmat[1, 0] = Dmat[0, 1]
        D[0, i] = Lz.T @ Dmat @ Lz
        D[1, i] = Lz.T @ Dmat @ Ly
        D[2, i] = Ly.T @ Dmat @ Ly

    # MESH
    numnodS = np.sum(Pr.nDivS * (nenS - 1)) + 1
    x = np.zeros(int(numnodS), dtype=mat_type)
    matelem = np.zeros(nelS, dtype=int)
    connS = np.full((np.max(nenS), nelS), -1, dtype=int)
    k, numnod = 0, 0
    for i in range(nL):
        steps_complex = CFEM_length(Pr.nDivS[i]) * Pr.h[i] if Pr.fem == 'cfem' else np.empty(())
        for j in range(Pr.nDivS[i]):
            matelem[k] = i
            connS[:nenS[i], k] = np.arange(numnod, numnod + nenS[i])
            b, e = connS[0, k], connS[nenS[i] - 1, k] + 1
            if Pr.fem == 'fem':
                x[b:e] = x[b] + np.arange(nenS[i]) * (Pr.h[i] / Pr.nDivS[i]) / (nenS[i] - 1)
            elif Pr.fem == 'cfem':
                x[b + 1:e] = x[b] + steps_complex[j]
            k, numnod = k + 1, numnod + nenS[i] - 1

    numnodT = numnodS if Pr.hsB == 'no' else numnodS + Pr.nDivB
    Kzz = np.zeros((2 * numnodT, 2 * numnodT), dtype=mat_type)
    Kzy = np.zeros((2 * numnodT, 2 * numnodT), dtype=mat_type)
    Kyy = np.zeros((2 * numnodT, 2 * numnodT), dtype=mat_type)
    M = np.zeros((2 * numnodT, 2 * numnodT), dtype=mat_type)

    j = 0
    for i in range(nL):
        GP = GPL[i]
        for k in range(Pr.nDivS[i]):
            j += 1
            Dzz = D[0, matelem[j - 1]]
            Dzy = D[1, matelem[j - 1]]
            Dyy = D[2, matelem[j - 1]]
            nde = connS[:nenS[i], j - 1]
            ndeE = np.hstack((2 * (nde + 1) - 1 - 1, 2 * (nde + 1) - 1)).reshape((-1, 1))
            ndeE = np.sort(ndeE.flatten())  # for elasticity 2 dof per node
            L = x[connS[nenS[i] - 1, j - 1]] - x[connS[0, j - 1]]
            detJ = L / 2
            for gpos in GP.T:
                s = gpos[0]
                wt = gpos[1]
                N, B = Nmat(s, nenS[i]), Bmat(s, nenS[i]) / detJ
                N, B = np.kron(N, np.eye(2)), np.kron(B, np.eye(2))
                Kzz[np.ix_(ndeE, ndeE)] += (N.T @ Dzz @ N) * (detJ * wt)
                Kzy[np.ix_(ndeE, ndeE)] -= (N.T @ Dzy @ B) * (detJ * wt)
                Kzy[np.ix_(ndeE, ndeE)] += (B.T @ Dzy.T @ N) * (detJ * wt)
                Kyy[np.ix_(ndeE, ndeE)] += (B.T @ Dyy @ B) * (detJ * wt)
                M[np.ix_(ndeE, ndeE)] += (N.T @ np.eye(2) @ N) * (detJ * wt * Pr.roS[matelem[j - 1]])

    # BOTTOM HALF-SPACE -------------------------------------------------------

    if Pr.hsB == 'yes':
        Dmat = np.zeros((3, 3))
        Dmat[0, 0] = Pr.roB * Pr.cpB ** 2
        Dmat[1, 1] = Dmat[0, 0]
        Dmat[2, 2] = Pr.roB * Pr.csB ** 2
        Dmat[0, 1] = Pr.roB * (Pr.cpB ** 2 - 2 * Pr.csB ** 2)
        Dmat[1, 0] = Dmat[0, 1]
        Dzz = Lz.T @ Dmat @ Lz
        Dzy = Lz.T @ Dmat @ Ly
        Dyy = Ly.T @ Dmat @ Ly
        nenB = 2
        nelB = Pr.nDivB
        numnodB = nelB * (nenB - 1) + 1
        connB = np.zeros((nenB, nelB), dtype=int)
        Lpmdl = (1) * Pr.pL1 * Pr.pAlp ** np.arange(0, nelB)
        # xFo = np.cumsum([0] + Lpmdl) + np.sum(Pr.h)

        for i in range(nenB):
            connB[i, 0:nelB] = np.arange(i, numnodB - (nenB - i) + 1, nenB - 1)

        connB += numnodS - 1
        GP = np.array([[0], [2]])

        for k in range(Pr.nDivB):
            nde = connB[:, k]
            ndeE = np.hstack((2 * (nde + 1) - 1 - 1, 2 * (nde + 1) - 1)).reshape((-1, 1))
            ndeE = np.sort(ndeE.flatten())
            L = Lpmdl[k]
            detJ = L / 2
            for gpos in range(GP.shape[1]):
                s = GP[0, gpos]
                wt = GP[1, gpos]
                N, B = Nmat(s, 2), Bmat(s, 2) / detJ
                N, B = np.kron(N, np.eye(2)), np.kron(B, np.eye(2))
                Kzz[np.ix_(ndeE, ndeE)] += (N.T @ Dzz @ N) * (detJ * wt)
                Kzy[np.ix_(ndeE, ndeE)] -= (N.T @ Dzy @ B) * (detJ * wt)
                Kzy[np.ix_(ndeE, ndeE)] += (B.T @ Dzy.T @ N) * (detJ * wt)
                Kyy[np.ix_(ndeE, ndeE)] += (B.T @ Dyy @ B) * (detJ * wt)
                M[np.ix_(ndeE, ndeE)] += (N.T @ np.eye(2) @ N) * (detJ * wt * Pr.roB)
        M = M[:-2, :-2]
        Kzz = Kzz[:-2, :-2]
        Kzy = Kzy[:-2, :-2]
        Kyy = Kyy[:-2, :-2]

    Z = np.zeros((Kzz.shape[0] // 2, Kzz.shape[0] // 2), dtype=mat_type)
    z = np.arange(0, Kzz.shape[0] - 1, 2)
    y = np.arange(1, Kzz.shape[0], 2)

    K2 = np.block([[Kzz[z][:, z], Z], [-Kzy[y][:, z], Kzz[y][:, y]]])
    K0 = np.block([[Kyy[z][:, z], Kzy[z][:, y]], [Z, Kyy[y][:, y]]])
    M = np.block([[M[z][:, z], Z], [Z, M[y][:, y]]])
    Kzz, Kzy, Kyy = None, None, None
    sizeS = K2.shape[0]
    print('Assembly: Done') if Pr.print == 'yes' else None

    # EIGENVALUE PROBLEM ------------------------------------------------------

    kz = np.zeros((sizeS, len(Pr.w)), dtype=complex)
    evcR = np.empty(())
    if Pr.cg == 'yes' or Pr.eff == 'yes':
        evcR = np.zeros((sizeS, sizeS, len(Pr.w)), dtype=complex)

    nrm = np.linalg.norm(M.flatten(), ord=1)
    K2 = K2 / nrm
    K0 = K0 / nrm
    M = M / nrm

    K0_M = np.hstack((K0, M))
    K0_M = np.linalg.solve(K2, K0_M)
    K2iK0 = K0_M[:, :K0.shape[1]]
    K2iM = K0_M[:, K0.shape[1]:]

    for i in range(len(Pr.w)):
        w = Pr.w[i] * 2 * np.pi
        if Pr.cg == 'yes' or Pr.eff == 'yes':
            kzi, evcRi = np.linalg.eig(-K2iK0 + w ** 2 * K2iM)
            # kzi, evcRi = sp.linalg.eig(K0 - w ** 2 * M, -K2)  # noBC/CFEM
            kzi = np.sqrt(kzi)
            evcR[:, :, i] = evcRi
        else:
            kzi = np.sqrt(np.linalg.eig(-K2iK0 + w ** 2 * K2iM)[0])
        kz[:sizeS, i] = kzi

    print('Eigen Solution: Done') if Pr.print == 'yes' else None

    # FILTERING ---------------------------------------------------------------

    for i in range(len(Pr.w)):
        kzi = kz[:, i]
        kzi[np.isinf(kzi)] = 0
        kzi[np.real(kzi) < 0] = 0
        kzi[np.abs(np.imag(kzi)) > Pr.kzTol] = 0
        kzi = np.real(kzi)
        if Pr.hsB == 'yes':
            w = Pr.w[i] * 2 * np.pi
            kzi[np.real(kzi) < w / Pr.csB] = 0
        kz[:, i] = kzi

    kz = np.real(kz)

    # GROUP VELOCITY -----------------------------------------------------------

    cg = np.empty(())
    if Pr.cg == 'yes':
        cg = np.zeros_like(kz)
        ne = evcR.shape[0] // 2
        for i in range(len(Pr.w)):
            w = Pr.w[i] * 2 * np.pi
            kzi = kz[:, i]
            iloc = np.where((kzi != 0) & (~np.isinf(kzi)) & (~np.isnan(kzi)))[0]
            if len(iloc) == 0:
                continue
            evcRi = evcR[:, iloc, i]
            evcLi = np.zeros_like(evcRi, dtype=complex)
            evcLi[:ne, :] = evcRi[:ne, :] @ np.diag(kzi[iloc])
            evcLi[ne:, :] = evcRi[ne:, :] @ np.diag(1. / kzi[iloc])
            for j in range(len(iloc)):
                cg[iloc[j], i] = np.real(-(evcLi[:, j].conj().T @ (2 * kzi[iloc[j]] * K2) @ evcRi[:, j]) /
                                         (evcLi[:, j].conj().T @ (-2 * w * M) @ evcRi[:, j]))
        print('Group Velocity: Done') if Pr.print == 'yes' else None

    # EFFECTIVE CURVE ---------------------------------------------------------
    cpE, cpEHR = np.empty(()), np.empty(())
    rNHR, padHR = int(2 ** 13), int(2 ** 14)  # High Resolution
    if Pr.eff == 'yes':
        cpE, cpEHR = np.zeros(len(Pr.w)), np.zeros(len(Pr.w))
        r = np.arange(Pr.r0, Pr.r0 + Pr.dr * (Pr.rN - 1) + 1e-10, Pr.dr)
        xmax = Pr.r0 + Pr.dr * (Pr.rN - 1)
        xmin = Pr.r0
        dx = Pr.dr
        dk = 2 * np.pi / (xmax - xmin)
        kmin = -np.pi / dx
        kmax = np.pi / dx
        k = np.arange(kmin, kmax - 1e-10, dk)

        rHR = np.arange(Pr.r0, Pr.r0 + Pr.dr * (rNHR - 1) + 1e-10, Pr.dr)
        xmaxHR = Pr.r0 + Pr.dr * (rNHR - 1)
        dkHR = 2 * np.pi / (xmaxHR - xmin)
        kHR = np.arange(kmin, kmax - 1e-10, dkHR)

        if Pr.pad != 0:
            xmax = Pr.r0 + (Pr.pad - 1) * Pr.dr
            dk = 2 * np.pi / (xmax - xmin)
            k = np.arange(kmin, kmax - 1e-10, dk)

        if padHR != 0:
            xmaxHR = Pr.r0 + (padHR - 1) * Pr.dr
            dkHR = 2 * np.pi / (xmaxHR - xmin)
            kHR = np.arange(kmin, kmax - 1e-10, dkHR)

        uxw = np.zeros(Pr.pad - 1, dtype=complex) if Pr.pad != 0 else np.zeros(Pr.rN, dtype=complex)
        uxwHR = np.zeros(padHR - 1, dtype=complex) if padHR != 0 else np.zeros(rNHR, dtype=complex)

        ne = evcR.shape[0] // 2

        for i in range(len(Pr.w)):
            w = Pr.w[i] * 2 * np.pi
            kzi = kz[:, i]
            if Pr.hsB == 'yes':
                kzi[np.real(kzi) < w / Pr.csB] = 0
            iloc = np.where((kzi != 0) & (~np.isinf(kzi)) & (~np.isnan(kzi)))[0]
            if len(iloc) == 0:
                cpE[i] = cpE[i - 1] if i > 0 else Pr.cpEffMax
                continue
            evcRi = evcR[:, iloc, i]
            evcLi = np.zeros_like(evcRi, dtype=complex)
            evcLi[:ne, :] = evcRi[:ne, :] * kzi[iloc]
            evcLi[ne:, :] = evcRi[ne:, :] * (1. / kzi[iloc])
            evcLiK2 = np.dot(evcLi.T, K2)
            nr = np.zeros(len(iloc), dtype=complex)
            for j in range(len(iloc)):
                nr[j] = np.dot(evcLiK2[j, :], evcRi[:, j]) / kzi[iloc[j]]
            evcRi_v = np.real(evcLi[ne, :] / np.sqrt(nr))

            kzin = kzi[iloc]
            bsl = sp.special.j1(kzin * Pr.R)
            uxw[:] = 0j
            uxwHR[:] = 0j

            for l in range(len(iloc)):
                kzin = kzi[iloc[l]]
                phi = evcRi_v[l] ** 2
                uxw[:Pr.rN] += phi * bsl[l] * sp.special.hankel1(0, kzin * r) * (
                        -1j * Pr.q * Pr.R * np.pi / (2 * kzin))
                # uxw[:Pr.rN] += phi * bsl[l] * np.sqrt(2 / (np.pi * kzin * r)) * np.exp(-1j * (kzin * r - np.pi / 4)) * (
                #         -1j * Pr.q * Pr.R * np.pi / (2 * kzin))  # approximate
                if Pr.interpOutlier == 'yes':
                    uxwHR[:rNHR] += phi * bsl[l] * sp.special.hankel1(0, kzin * rHR) * (
                            -1j * Pr.q * Pr.R * np.pi / (2 * kzin))

            uxw[Pr.rN - 1] = 0j
            ukw = np.abs(np.fft.fftshift(np.fft.fft(uxw)))
            kE = np.argmax(ukw)
            cpE[i] = w / np.abs(k[kE])
            if cpE[i] > Pr.cpEffMax:
                cpE[i] = cpE[i - 1] if i > 0 else Pr.cpEffMax

            if Pr.interpOutlier == 'yes':
                uxwHR[rNHR - 1] = 0j
                ukwHR = np.abs(np.fft.fftshift(np.fft.fft(uxwHR)))
                kEHR = np.argmax(ukwHR)
                cpEHR[i] = w / np.abs(kHR[kEHR])
                if cpEHR[i] > Pr.cpEffMax:
                    cpEHR[i] = cpEHR[i - 1] if i > 0 else Pr.cpEffMax
        print('Effective Curve: Done') if Pr.print == 'yes' else None

    # TRACING  -------------------------------------------------------------------

    cgT, cpT, kzT = [], [], []
    cph = np.zeros_like(kz)
    if trace == 'yes':
        # cph = np.where(kz != 0, np.divide(np.tile(Pr.w * 2 * np.pi, (kz.shape[0], 1)), kz), np.inf)
        cph[:] = 0.
        cph[kz != 0] = np.tile(Pr.w * 2 * np.pi, (kz.shape[0], 1))[kz != 0] / kz[kz != 0]
        cph[kz == 0] = np.inf
        for i in range(nMode):
            cpm, cpl = np.min(cph, axis=0), np.argmin(cph, axis=0)
            k = kz[cpl, np.arange(len(cpl))]
            iloc = np.where(cpm == np.inf)[0]
            cpm[iloc] = -1
            k[iloc] = 0
            cpT.append(cpm)
            if Pr.cg == 'yes':
                cg_temp = np.zeros(cph.shape[1])
                for j in range(cph.shape[1]):
                    cg_temp[j] = cg[cpl[j], j]
                cgT.append(cg_temp)
            kzT.append([kz[cpl[j], j] for j in range(len(cpl))])
            for j in range(len(cpl)):
                cph[cpl[j], j] = np.inf
        cpT = cpT[:min(nMode, len(cpT))]
        print('Tracing: Done') if Pr.print == 'yes' and Pr.trace == 'yes' else None

        # OUTLIERS  ------------------------------------------------------------------

        if Pr.eff == 'yes' and Pr.interpOutlier == 'yes':
            modeHR = np.argmin(np.abs(np.vstack(cpT) - cpEHR[np.newaxis, :]), axis=0)

            # dominant first mode
            if modeHR[0] == modeHR[-1]:
                cpE = cpT[modeHR[0]]
                if Pr.trace == 'no':
                    cgT, cpT, kzT = [], [], []
                print('Outliers: Done') if Pr.print == 'yes' else None
                return kz, cg, cpE, cgT, cpT, kzT

            invalid_entries = []
            for i in range(len(modeHR)):
                if any(modeHR[i + 1:] < modeHR[i]):
                    invalid_entries.append(i)
            invalid_entries = np.array(invalid_entries)

            # first mode
            last_zero = np.where(modeHR == 0)[0][-1]
            cpE[:last_zero + 1] = cpT[0][:last_zero + 1]
            invalid_entries = invalid_entries[invalid_entries > last_zero]

            # find outliers (1st pass)
            threshold = 1
            window_size = len(cpE) // 1
            bg = -1  # last_zero (only higher modes), -1 (entire cpE)
            cpEr = cpE[bg + 1:]
            if len(cpEr) > 0:
                for i in range(len(cpEr) - window_size + 1):
                    window = cpEr[i:i + window_size]
                    z_scores = np.abs((window - window.mean()) / window.std())
                    window[z_scores >= threshold] = np.nan
                    cpEr[i:i + window_size] = window
                cpE[bg + 1:] = cpEr

            if len(invalid_entries) > 0:
                cpE[invalid_entries] = np.nan

            # first mode
            cpE[:last_zero + 1] = cpT[0][:last_zero + 1]
            # other modes
            nan_indices, nan_regions = np.where(np.isnan(cpE))[0], []
            if len(nan_indices) > 0:
                start_idx, prev_idx = nan_indices[0], nan_indices[0]
                for idx in nan_indices[1:]:
                    if idx != prev_idx + 1:
                        nan_regions.append((start_idx, prev_idx))
                        start_idx = idx
                    prev_idx = idx
                nan_regions.append((start_idx, prev_idx))
                for region in nan_regions:
                    start_idx, end_idx = region
                    cpT_elems = np.array([array[start_idx - 1] for array in cpT])
                    closest_b = np.argmin(np.abs(cpT_elems - cpE[start_idx - 1]))
                    cpE[start_idx:end_idx + 1] = cpT[closest_b][start_idx:end_idx + 1]

            # find outliers (2nd pass)
            threshold = 1
            cpEr = cpE[last_zero + 1:]
            if len(cpEr) > 0:
                window_size = len(cpEr) // 1
                cpE_one_pass = cpE.copy()
                for i in range(len(cpEr) - window_size + 1):
                    window = cpEr[i:i + window_size]
                    z_scores = np.abs((window - window.mean()) / window.std())
                    window[z_scores >= threshold] = np.nan
                    cpEr[i:i + window_size] = window
                cpE[last_zero + 1:] = cpEr

                # replace NaNs at the end to avoid interpolation issues
                if np.isnan(cpE[-1]):
                    last_non_nan = len(cpE) - np.where(~np.isnan(cpE[::-1]))[0][0] - 1
                    cpE[last_non_nan + 1:] = cpE_one_pass[last_non_nan + 1:]

                # interpolate
                method_1, method_2 = 'quadratic', 'linear'  # zero, linear, quadratic, ...
                min_bef, max_bef, cpE_nan = np.nanmin(cpE), np.nanmax(cpE), cpE.copy()
                nan_indices = np.isnan(cpE)
                not_nan_indices = ~nan_indices
                x = np.arange(len(cpE))
                try:
                    interp_func = sp.interpolate.interp1d(x[not_nan_indices], cpE[not_nan_indices], kind=method_1)
                    cpE[nan_indices] = interp_func(x[nan_indices])
                    cpE[:last_zero + 1] = cpT[0][:last_zero + 1]
                    min_aft, max_aft = np.nanmin(cpE), np.nanmax(cpE)
                    if min_aft < min_bef or max_bef < max_aft:
                        raise Exception
                except Exception:
                    try:
                        interp_func = sp.interpolate.interp1d(x[not_nan_indices], cpE_nan[not_nan_indices],
                                                              kind=method_2)
                        cpE_nan[nan_indices] = interp_func(x[nan_indices])
                        cpE_nan[:last_zero + 1] = cpT[0][:last_zero + 1]
                        cpE = cpE_nan
                    except Exception:
                        print('One pass is used for outliers')
                        cpE = cpE_one_pass
                        cpE[:last_zero + 1] = cpT[0][:last_zero + 1]

            if Pr.trace == 'no':
                cgT, cpT, kzT = [], [], []
            print('Outliers: Done') if Pr.print == 'yes' else None

    return kz, cg, cpE, cgT, cpT, kzT

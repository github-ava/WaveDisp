import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.Utilities import Parameters
from src.WaveDisp import WaveDisp
from src.ForwardProb import ForwardHS_First, ForwardHS_Eff


def test_forward():
    nu = 0.35
    cp_cs_fac = np.sqrt((2 - 2 * nu) / (1 - 2 * nu))
    # --------------
    Pr = Parameters()

    # SOLID
    # -----
    # Pr.cs = np.array([200., 300., 400.])  # Layer Shear Wave Velocity (m/s)
    Pr.cs = np.array([300., 200., 400.])  # Layer Shear Wave Velocity (m/s)
    # Pr.cs = np.array([300., 400., 200.])  # Layer Shear Wave Velocity (m/s)
    # Pr.cs = np.array([300., 200., 300.])  # Layer Shear Wave Velocity (m/s)
    # -----
    Pr.cp = Pr.cs * cp_cs_fac  # Layer Pressure Wave Velocity (m/s)
    Pr.roS = np.array([1800., 1800., 1800.])  # Layer Density (kg/m3)
    Pr.h = np.array([6., 4., 8.])  # Layer Thickness (m)
    Pr.nDivS = np.array([1, 1, 1], dtype=int)  # Number of Elements per Layer
    Pr.fem = 'fem'  # Method: fem or cfem (Complex-FEM)
    Pr.order = np.array([10, 10, 10], dtype=int)  # FEM order 1, 2, 3, ..., 10

    # BOTTOM HALF-SPACE [using PMDLs]
    Pr.hsB = 'yes'  # Bottom Half-Space 'yes','no'
    Pr.csB = 500.  # HS Shear Wave Velocity (m/s)
    Pr.cpB = Pr.csB * cp_cs_fac  # HS Pressure Wave Velocity (m/s)
    Pr.roB = 1800.  # HS Density (kg/m3)
    Pr.nDivB = 10  # Number of (linear) PMDL elements
    Pr.pL1 = 1.  # 1st Layer Length: Lj = L1 * alpha ^ (j-1)
    Pr.pAlp = 2.  # Increase Ratio: Lj = L1 * alpha ^ (j-1)

    # EFFECTIVE DISPERSION CURVE
    Pr.eff = 'yes'  # Effective Dispersion Curve Calculation: 'yes','no'
    Pr.r0 = 20.  # Minimum Receiver Offset (Acquisition Layout)
    Pr.dr = 0.5  # Minimum Receiver Distance (Acquisition Layout)
    Pr.rN = 60  # Number of Receivers (Acquisition Layout)
    Pr.pad = int(2 ** 14)  # Number of Padding Layers in FFT Calculation (0: No Padding)
    Pr.q = 1.e6  # Magnitude of Circular Distributed load (N/m2)
    Pr.R = 1.e-2  # Radius of Circular Distributed load (m)
    Pr.cpEffMax = 1500.  # Maximum Effective Phase Velocity (m/s)
    Pr.interpOutlier = 'yes'  # interpolate the outlier values in effective curve

    # DISPLAY SETTINGS
    Pr.w = np.arange(3., 80.5, 0.5)  # Frequency (Hz)
    Pr.cg = 'yes'  # Group Velocity Calculation: 'yes','no'
    Pr.kzTol = 1.e-3  # Filtering Tolerance for imag(eigenvalue)
    Pr.cpMax = 490.  # Maximum Phase Velocity (m/s)
    Pr.cpMin = 150.  # Minimum Phase Velocity (m/s)
    Pr.trace = 'yes'  # Trace Curves: 'yes','no'
    Pr.nMode = 10  # Number of Modes to Trace

    # PRINT
    Pr.print = 'yes'  # Print info: 'yes','no'

    t = time()
    kz, cg, cpE, cgT, cpT, kzT = WaveDisp(Pr)
    print('time: {:.3f} s'.format(time() - t))
    # return

    if Pr.trace == 'no':
        plt.figure(1)
        for i in range(kz.shape[0]):
            iloc = np.where(kz[i, :] != 0)[0]
            pT = plt.plot(Pr.w[iloc], 2 * np.pi * Pr.w[iloc] / np.real(kz[i, iloc]), 'b.')
        if Pr.eff == 'yes':
            iloc = np.where(cpE != 0)[0]
            plt.plot(Pr.w[iloc], cpE[iloc], 'ro', markerfacecolor='none')
            plt.title('B: Theoretical    R: Effective')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase Velocity (m/s)')
        plt.ylim([Pr.cpMin, Pr.cpMax])
        plt.xlim(0, max(Pr.w))
        plt.box(True)

        plt.figure(2)
        for i in range(kz.shape[0]):
            iloc = np.where(kz[i, :] != 0)[0]
            plt.plot(Pr.w[iloc], np.real(kz[i, iloc]), 'k.')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavenumber (rad/m)')
        plt.xlim(0, max(Pr.w))
        plt.box(True)

        if Pr.cg == 'yes':
            plt.figure(3)
            for i in range(kz.shape[0]):
                ilocK = np.where(kz[i, :] != 0)[0]
                ilocG = np.where(cg[i, :] != 0)[0]
                iloc = np.intersect1d(ilocK, ilocG)
                plt.plot(Pr.w[iloc], cg[i, iloc], 'r.')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Group Velocity (m/s)')
            plt.xlim(0, max(Pr.w))
            plt.box(True)

        plt.show()

    elif Pr.trace == 'yes':

        c_l = ['k', 'r', 'b', 'g', 'm'] * 20

        plt.figure(1)
        for i in range(len(cpT)):
            w, c = Pr.w, cpT[i]
            iloc = np.where(c > 0)[0]
            c, w = c[iloc], w[iloc]
            if Pr.eff == 'no':
                plt.plot(w, c, '-', color=c_l[i], linewidth=1.5)
            elif Pr.eff == 'yes':
                if i == 0:
                    pT = plt.plot(w, c, '-', color='b', linewidth=1.5, label='Theoretical')
                else:
                    pT = plt.plot(w, c, '-', color='b', linewidth=1.5)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase Velocity (m/s)')
            plt.ylim([Pr.cpMin, Pr.cpMax])
            plt.xlim(0, max(Pr.w))
            plt.box(True)

        if Pr.eff == 'yes':
            iloc = np.where(cpE != 0)[0]
            pE = plt.plot(Pr.w[iloc], cpE[iloc], 'ro', markerfacecolor='none', label='Effective')
            plt.legend()
        plt.figure(2)
        for i in range(len(kzT)):
            w, k = Pr.w, np.array(kzT[i])
            iloc = np.where(k != 0)[0]
            k, w = k[iloc], w[iloc]
            plt.plot(w, np.real(k), '-', color=c_l[i], linewidth=1.5)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Wavenumber (rad/m)')
            plt.xlim(0, max(Pr.w))
            plt.box(True)

        if Pr.cg == 'yes':
            plt.figure(3)
            for i in range(len(cpT)):
                w, c, g = Pr.w, cpT[i], cgT[i]
                ilocC = np.where(c > 0)[0]
                ilocG = np.where(g > 0)[0]
                iloc = np.intersect1d(ilocC, ilocG)
                c, w, g = c[iloc], w[iloc], g[iloc]
                plt.plot(w, g, '-', color=c_l[i], linewidth=1.5)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Group Velocity (m/s)')
                plt.xlim(0, max(Pr.w))
                plt.box(True)

        plt.show()


def test_forward_first_mode():
    params = np.array([300., 200., 400., 500., 6., 4., 8.])
    w, cp = ForwardHS_First(params)

    plt.plot(w, cp, 'o', color='b', linewidth=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Velocity (m/s)')
    plt.xlim(0, max(w))
    plt.box(True)
    plt.show()


def test_forward_eff_curve():
    params = np.array([300., 200., 400., 500., 6., 4., 8.])
    w, cpE = ForwardHS_Eff(params)

    plt.plot(w, cpE, 'o', color='b', linewidth=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Effective Phase Velocity (m/s)')
    plt.xlim(0, max(w))
    plt.box(True)
    plt.show()


if __name__ == "__main__":
    test_forward()
    # test_forward_first_mode()
    # test_forward_eff_curve()

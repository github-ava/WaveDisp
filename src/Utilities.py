import numpy as np
import scipy as sp


class Parameters:
    # SOLID
    cs = np.array([300., 200., 400.])  # Layer Shear Wave Velocity (m/s)
    cp = np.array([600., 400., 800.])  # Layer Pressure Wave Velocity (m/s)
    roS = np.array([1800., 1800., 1800.])  # Layer Density (kg/m3)
    h = np.array([6., 4., 8.])  # Layer Thickness (m)
    nDivS = np.array([1, 1, 1], dtype=int)  # Number of Elements per Layer
    fem = 'fem'  # Method: fem or cfem (Complex-FEM)
    order = np.array([6, 6, 6], dtype=int)  # FEM order 1, 2, 3, ..., 10

    # BOTTOM HALF-SPACE [using PMDLs]
    hsB = 'yes'  # Bottom Half-Space 'yes','no'
    csB = 500.  # HS Shear Wave Velocity (m/s)
    cpB = 1000.  # HS Pressure Wave Velocity (m/s)
    roB = 1800.  # HS Density (kg/m3)
    nDivB = 10  # Number of (linear) PMDL elements
    pL1 = 1.  # 1st Layer Length: Lj = L1 * alpha ** (j-1)
    pAlp = 2.  # Increase Ratio: Lj = L1 * alpha ** (j-1)

    # EFFECTIVE DISPERSION CURVE
    eff = 'yes'  # Effective Dispersion Curve Calculation: 'yes','no'
    r0 = 20.  # Minimum Receiver Offset (Acquisition Layout)
    dr = 0.5  # Minimum Receiver Distance (Acquisition Layout)
    rN = 60  # Number of Receivers (Acquisition Layout)
    pad = int(2 ** 14)  # Number of Padding Layers in FFT Calculation (0: No Padding)
    q = 1.e6  # Magnitude of Circular Distributed load (N/m2)
    R = 1.e-2  # Radius of Circular Distributed load (m)
    cpEffMax = 1500.  # Maximum Effective Phase Velocity (m/s)
    interpOutlier = 'yes'  # interpolate the outlier values in effective curve

    # DISPLAY SETTINGS
    w = np.arange(3., 80.5, 0.5)  # Frequency (Hz)
    cg = 'yes'  # Group Velocity Calculation: 'yes','no'
    kzTol = 1.e-3  # Filtering Tolerance for imag(eigenvalue)
    cpMax = 490.  # Maximum Phase Velocity (m/s)
    cpMin = 150.  # Minimum Phase Velocity (m/s)
    trace = 'yes'  # Trace Curves: 'yes','no'
    nMode = 10  # Number of Modes to Trace

    # PRINT
    print = 'yes'  # Print info: 'yes','no'


ParametersNN = {
    'expr_id': 0,
    'in_dir': '../data/data_test',
    'out_dir': '../out/out_test',
    'metrics': 0,  # 0: ['mean_squared_error']
    'monitor': 0,  # 0: MinMaxScaler
    'scaling': 0,  # 0: MinMaxScaler
    'loss': 2,  # 0: MSE, 1: MSLE, 2: MAE, 3: Huber
    'model': 0,  # 0: ANN , 0: CNN
    'optimizer': 1,  # 0: Adam, 1: Nadam
    'initializer': 2,  # 0: he_normal, 1:glorot_normal, 2: random_normal
    'activation': 3,  # 0: relu, 1: sigmoid, 2: LeakyReLU, 3: custom_leaky_relu
    'batch_size': int(2 ** 7),
    'learning_rate': 1.e-4,
    'train_ratio': 0.80,
    'val_ratio': 0.15,
    'max_epochs': 800,
    'save_period': 10000,
    'patience': 1000,
    'num_samples': 0,  # 0: all samples
    'h_fixed': -1.,  # -1.: Cp/H, value: Cp with fixed H=value
    'verbose': 0,  # training 0,1,2
    'save_model': False,
    'shuffle_data': False,
    'show_plots': False,
    'training': True,
}


def Nmat(x: float, n: int) -> np.ndarray:
    """
    Compute the FEM shape functions

    Parameters:
        x (float): The coordinate value.
        n (int): The number of nodes.

    Returns:
        np.ndarray: Array of shape functions for the given coordinate and number of nodes.
    """
    if n == 2:
        N = np.array([1 / 2 * (1 - x), 1 / 2 * (1 + x)])
    elif n == 3:
        N = np.array([-1 / 2 * x * (1 - x), (1 - x ** 2), 1 / 2 * x * (1 + x)])
    elif n == 4:
        N = np.array([9 / 16 * (1 - x) * (x ** 2 - 1 / 9), 27 / 16 * (x ** 2 - 1) * (x - 1 / 3),
                      27 / 16 * (1 - x ** 2) * (x + 1 / 3), 9 / 16 * (1 + x) * (x ** 2 - 1 / 9)])
    elif n == 5:
        N = np.array(
            [-(x * (- 4 * x ** 3 + 4 * x ** 2 + x - 1)) / 6, (4 * x * (- 2 * x ** 3 + x ** 2 + 2 * x - 1)) / 3,
             4 * x ** 4 - 5 * x ** 2 + 1, (4 * x * (- 2 * x ** 3 - x ** 2 + 2 * x + 1)) / 3,
             -(x * (- 4 * x ** 3 - 4 * x ** 2 + x + 1)) / 6])
    elif n == 6:
        N = np.array([- (625 * x ** 5) / 768 + (625 * x ** 4) / 768 + (125 * x ** 3) / 384 - (125 * x ** 2) / 384 - (
                3 * x) / 256 + 3 / 256,
                      (3125 * x ** 5) / 768 - (625 * x ** 4) / 256 - (1625 * x ** 3) / 384 + (325 * x ** 2) / 128 + (
                              125 * x) / 768 - 25 / 256,
                      - (3125 * x ** 5) / 384 + (625 * x ** 4) / 384 + (2125 * x ** 3) / 192 - (425 * x ** 2) / 192 - (
                              375 * x) / 128 + 75 / 128,
                      (3125 * x ** 5) / 384 + (625 * x ** 4) / 384 - (2125 * x ** 3) / 192 - (425 * x ** 2) / 192 + (
                              375 * x) / 128 + 75 / 128,
                      - (3125 * x ** 5) / 768 - (625 * x ** 4) / 256 + (1625 * x ** 3) / 384 + (325 * x ** 2) / 128 - (
                              125 * x) / 768 - 25 / 256,
                      (625 * x ** 5) / 768 + (625 * x ** 4) / 768 - (125 * x ** 3) / 384 - (125 * x ** 2) / 384 + (
                              3 * x) / 256 + 3 / 256])
    elif n == 7:
        N = np.array([(x * (81 * x ** 5 - 81 * x ** 4 - 45 * x ** 3 + 45 * x ** 2 + 4 * x - 4)) / 80,
                      -(9 * x * (27 * x ** 5 - 18 * x ** 4 - 30 * x ** 3 + 20 * x ** 2 + 3 * x - 2)) / 40,
                      (9 * x * (27 * x ** 5 - 9 * x ** 4 - 39 * x ** 3 + 13 * x ** 2 + 12 * x - 4)) / 16,
                      - (81 * x ** 6) / 4 + (63 * x ** 4) / 2 - (49 * x ** 2) / 4 + 1,
                      (9 * x * (27 * x ** 5 + 9 * x ** 4 - 39 * x ** 3 - 13 * x ** 2 + 12 * x + 4)) / 16,
                      -(9 * x * (27 * x ** 5 + 18 * x ** 4 - 30 * x ** 3 - 20 * x ** 2 + 3 * x + 2)) / 40,
                      (x * (81 * x ** 5 + 81 * x ** 4 - 45 * x ** 3 - 45 * x ** 2 + 4 * x + 4)) / 80])
    elif n == 8:
        N = np.array([- (117649 * x ** 7) / 92160 + (117649 * x ** 6) / 92160 + (16807 * x ** 5) / 18432 - (
                16807 * x ** 4) / 18432 - (12691 * x ** 3) / 92160 + (12691 * x ** 2) / 92160 + (
                              5 * x) / 2048 - 5 / 2048,
                      (823543 * x ** 7) / 92160 - (117649 * x ** 6) / 18432 - (991613 * x ** 5) / 92160 + (
                              141659 * x ** 4) / 18432 + (171157 * x ** 3) / 92160 - (24451 * x ** 2) / 18432 - (
                              343 * x) / 10240 + 49 / 2048,
                      - (823543 * x ** 7) / 30720 + (117649 * x ** 6) / 10240 + (84035 * x ** 5) / 2048 - (
                              36015 * x ** 4) / 2048 - (148519 * x ** 3) / 10240 + (63651 * x ** 2) / 10240 + (
                              1715 * x) / 6144 - 245 / 2048,
                      (823543 * x ** 7) / 18432 - (117649 * x ** 6) / 18432 - (1394981 * x ** 5) / 18432 + (
                              199283 * x ** 4) / 18432 + (648613 * x ** 3) / 18432 - (92659 * x ** 2) / 18432 - (
                              8575 * x) / 2048 + 1225 / 2048,
                      - (823543 * x ** 7) / 18432 - (117649 * x ** 6) / 18432 + (1394981 * x ** 5) / 18432 + (
                              199283 * x ** 4) / 18432 - (648613 * x ** 3) / 18432 - (92659 * x ** 2) / 18432 + (
                              8575 * x) / 2048 + 1225 / 2048,
                      (823543 * x ** 7) / 30720 + (117649 * x ** 6) / 10240 - (84035 * x ** 5) / 2048 - (
                              36015 * x ** 4) / 2048 + (148519 * x ** 3) / 10240 + (63651 * x ** 2) / 10240 - (
                              1715 * x) / 6144 - 245 / 2048,
                      - (823543 * x ** 7) / 92160 - (117649 * x ** 6) / 18432 + (991613 * x ** 5) / 92160 + (
                              141659 * x ** 4) / 18432 - (171157 * x ** 3) / 92160 - (24451 * x ** 2) / 18432 + (
                              343 * x) / 10240 + 49 / 2048,
                      (117649 * x ** 7) / 92160 + (117649 * x ** 6) / 92160 - (16807 * x ** 5) / 18432 - (
                              16807 * x ** 4) / 18432 + (12691 * x ** 3) / 92160 + (12691 * x ** 2) / 92160 - (
                              5 * x) / 2048 - 5 / 2048])
    elif n == 9:
        N = np.array([-(x * (
                - 1024 * x ** 7 + 1024 * x ** 6 + 896 * x ** 5 - 896 * x ** 4 - 196 * x ** 3 + 196 * x ** 2 + 9 * x - 9)) / 630,
                      (16 * x * (
                              - 256 * x ** 7 + 192 * x ** 6 + 336 * x ** 5 - 252 * x ** 4 - 84 * x ** 3 + 63 * x ** 2 + 4 * x - 3)) / 315,
                      -(4 * x * (
                              - 512 * x ** 7 + 256 * x ** 6 + 832 * x ** 5 - 416 * x ** 4 - 338 * x ** 3 + 169 * x ** 2 + 18 * x - 9)) / 45,
                      (16 * x * (
                              - 256 * x ** 7 + 64 * x ** 6 + 464 * x ** 5 - 116 * x ** 4 - 244 * x ** 3 + 61 * x ** 2 + 36 * x - 9)) / 45,
                      (1024 * x ** 8) / 9 - (640 * x ** 6) / 3 + (364 * x ** 4) / 3 - (205 * x ** 2) / 9 + 1, (
                              16 * x * (
                              - 256 * x ** 7 - 64 * x ** 6 + 464 * x ** 5 + 116 * x ** 4 - 244 * x ** 3 - 61 * x ** 2 + 36 * x + 9)) / 45,
                      -(4 * x * (
                              - 512 * x ** 7 - 256 * x ** 6 + 832 * x ** 5 + 416 * x ** 4 - 338 * x ** 3 - 169 * x ** 2 + 18 * x + 9)) / 45,
                      (16 * x * (
                              - 256 * x ** 7 - 192 * x ** 6 + 336 * x ** 5 + 252 * x ** 4 - 84 * x ** 3 - 63 * x ** 2 + 4 * x + 3)) / 315,
                      -(x * (
                              - 1024 * x ** 7 - 1024 * x ** 6 + 896 * x ** 5 + 896 * x ** 4 - 196 * x ** 3 - 196 * x ** 2 + 9 * x + 9)) / 630])
    elif n == 10:
        N = np.array([- (4782969 * x ** 9) / 2293760 + (4782969 * x ** 8) / 2293760 + (177147 * x ** 7) / 81920 - (
                177147 * x ** 6) / 81920 - (102789 * x ** 5) / 163840 + (102789 * x ** 4) / 163840 + (
                              29061 * x ** 3) / 573440 - (29061 * x ** 2) / 573440 - (35 * x) / 65536 + 35 / 65536,
                      (43046721 * x ** 9) / 2293760 - (4782969 * x ** 8) / 327680 - (15411789 * x ** 7) / 573440 + (
                              1712421 * x ** 6) / 81920 + (1449981 * x ** 5) / 163840 - (
                              1127763 * x ** 4) / 163840 - (429381 * x ** 3) / 573440 + (47709 * x ** 2) / 81920 + (
                              3645 * x) / 458752 - 405 / 65536,
                      - (43046721 * x ** 9) / 573440 + (4782969 * x ** 8) / 114688 + (531441 * x ** 7) / 4096 - (
                              295245 * x ** 6) / 4096 - (2473497 * x ** 5) / 40960 + (274833 * x ** 4) / 8192 + (
                              165483 * x ** 3) / 28672 - (91935 * x ** 2) / 28672 - (
                              5103 * x) / 81920 + 567 / 16384,
                      (14348907 * x ** 9) / 81920 - (4782969 * x ** 8) / 81920 - (6908733 * x ** 7) / 20480 + (
                              2302911 * x ** 6) / 20480 + (8063469 * x ** 5) / 40960 - (
                              2687823 * x ** 4) / 40960 - (718497 * x ** 3) / 20480 + (239499 * x ** 2) / 20480 + (
                              6615 * x) / 16384 - 2205 / 16384,
                      - (43046721 * x ** 9) / 163840 + (4782969 * x ** 8) / 163840 + (21789081 * x ** 7) / 40960 - (
                              2421009 * x ** 6) / 40960 - (28258227 * x ** 5) / 81920 + (
                              3139803 * x ** 4) / 81920 + (3324969 * x ** 3) / 40960 - (369441 * x ** 2) / 40960 - (
                              178605 * x) / 32768 + 19845 / 32768,
                      (43046721 * x ** 9) / 163840 + (4782969 * x ** 8) / 163840 - (21789081 * x ** 7) / 40960 - (
                              2421009 * x ** 6) / 40960 + (28258227 * x ** 5) / 81920 + (
                              3139803 * x ** 4) / 81920 - (3324969 * x ** 3) / 40960 - (369441 * x ** 2) / 40960 + (
                              178605 * x) / 32768 + 19845 / 32768,
                      - (14348907 * x ** 9) / 81920 - (4782969 * x ** 8) / 81920 + (6908733 * x ** 7) / 20480 + (
                              2302911 * x ** 6) / 20480 - (8063469 * x ** 5) / 40960 - (
                              2687823 * x ** 4) / 40960 + (718497 * x ** 3) / 20480 + (239499 * x ** 2) / 20480 - (
                              6615 * x) / 16384 - 2205 / 16384,
                      (43046721 * x ** 9) / 573440 + (4782969 * x ** 8) / 114688 - (531441 * x ** 7) / 4096 - (
                              295245 * x ** 6) / 4096 + (2473497 * x ** 5) / 40960 + (274833 * x ** 4) / 8192 - (
                              165483 * x ** 3) / 28672 - (91935 * x ** 2) / 28672 + (
                              5103 * x) / 81920 + 567 / 16384,
                      - (43046721 * x ** 9) / 2293760 - (4782969 * x ** 8) / 327680 + (15411789 * x ** 7) / 573440 + (
                              1712421 * x ** 6) / 81920 - (1449981 * x ** 5) / 163840 - (
                              1127763 * x ** 4) / 163840 + (429381 * x ** 3) / 573440 + (47709 * x ** 2) / 81920 - (
                              3645 * x) / 458752 - 405 / 65536,
                      (4782969 * x ** 9) / 2293760 + (4782969 * x ** 8) / 2293760 - (177147 * x ** 7) / 81920 - (
                              177147 * x ** 6) / 81920 + (102789 * x ** 5) / 163840 + (102789 * x ** 4) / 163840 - (
                              29061 * x ** 3) / 573440 - (29061 * x ** 2) / 573440 + (35 * x) / 65536 + 35 / 65536])
    elif n == 11:
        N = np.array([(x * (
                390625 * x ** 9 - 390625 * x ** 8 - 468750 * x ** 7 + 468750 * x ** 6 + 170625 * x ** 5 - 170625 * x ** 4 - 20500 * x ** 3 + 20500 * x ** 2 + 576 * x - 576)) / 145152,
                      -(25 * x * (
                              78125 * x ** 9 - 62500 * x ** 8 - 121875 * x ** 7 + 97500 * x ** 6 + 49875 * x ** 5 - 39900 * x ** 4 - 6305 * x ** 3 + 5044 * x ** 2 + 180 * x - 144)) / 72576,
                      (25 * x * (
                              78125 * x ** 9 - 46875 * x ** 8 - 143750 * x ** 7 + 86250 * x ** 6 + 76125 * x ** 5 - 45675 * x ** 4 - 10820 * x ** 3 + 6492 * x ** 2 + 320 * x - 192)) / 16128,
                      -(25 * x * (
                              78125 * x ** 9 - 31250 * x ** 8 - 159375 * x ** 7 + 63750 * x ** 6 + 102375 * x ** 5 - 40950 * x ** 4 - 21845 * x ** 3 + 8738 * x ** 2 + 720 * x - 288)) / 6048,
                      (25 * x * (
                              78125 * x ** 9 - 15625 * x ** 8 - 168750 * x ** 7 + 33750 * x ** 6 + 121125 * x ** 5 - 24225 * x ** 4 - 33380 * x ** 3 + 6676 * x ** 2 + 2880 * x - 576)) / 3456,
                      - (390625 * x ** 10) / 576 + (859375 * x ** 8) / 576 - (213125 * x ** 6) / 192 + (
                              191125 * x ** 4) / 576 - (5269 * x ** 2) / 144 + 1, (25 * x * (
                    78125 * x ** 9 + 15625 * x ** 8 - 168750 * x ** 7 - 33750 * x ** 6 + 121125 * x ** 5 + 24225 * x ** 4 - 33380 * x ** 3 - 6676 * x ** 2 + 2880 * x + 576)) / 3456,
                      -(25 * x * (
                              78125 * x ** 9 + 31250 * x ** 8 - 159375 * x ** 7 - 63750 * x ** 6 + 102375 * x ** 5 + 40950 * x ** 4 - 21845 * x ** 3 - 8738 * x ** 2 + 720 * x + 288)) / 6048,
                      (25 * x * (
                              78125 * x ** 9 + 46875 * x ** 8 - 143750 * x ** 7 - 86250 * x ** 6 + 76125 * x ** 5 + 45675 * x ** 4 - 10820 * x ** 3 - 6492 * x ** 2 + 320 * x + 192)) / 16128,
                      -(25 * x * (
                              78125 * x ** 9 + 62500 * x ** 8 - 121875 * x ** 7 - 97500 * x ** 6 + 49875 * x ** 5 + 39900 * x ** 4 - 6305 * x ** 3 - 5044 * x ** 2 + 180 * x + 144)) / 72576,
                      (x * (
                              390625 * x ** 9 + 390625 * x ** 8 - 468750 * x ** 7 - 468750 * x ** 6 + 170625 * x ** 5 + 170625 * x ** 4 - 20500 * x ** 3 - 20500 * x ** 2 + 576 * x + 576)) / 145152])
    else:
        N = np.empty(())
    return N


def Bmat(x: float, n: int) -> np.ndarray:
    """
    Compute the derivative of FEM shape functions

    Parameters:
        x (float): The coordinate value.
        n (int): The number of nodes.

    Returns:
        np.ndarray: Array of derivatives of shape functions for the given coordinate and number of nodes.
    """
    if n == 2:
        B = np.array([-1 / 2, 1 / 2])
    elif n == 3:
        B = np.array([-1 / 2 * (1 - 2 * x), (-2 * x), 1 / 2 * (1 + 2 * x)])
    elif n == 4:
        B = np.array([(9 * x) / 8 - (27 * x ** 2) / 16 + 1 / 16, (81 * x ** 2) / 16 - (9 * x) / 8 - 27 / 16,
                      27 / 16 - (81 * x ** 2) / 16 - (9 * x) / 8, (27 * x ** 2) / 16 + (9 * x) / 8 - 1 / 16])
    elif n == 5:
        B = np.array([(8 * x ** 3) / 3 - 2 * x ** 2 - x / 3 + 1 / 6,
                      - (32 * x ** 3) / 3 + 4 * x ** 2 + (16 * x) / 3 - 4 / 3,
                      16 * x ** 3 - 10 * x, - (32 * x ** 3) / 3 - 4 * x ** 2 + (16 * x) / 3 + 4 / 3,
                      (8 * x ** 3) / 3 + 2 * x ** 2 - x / 3 - 1 / 6])
    elif n == 6:
        B = np.array([- (3125 * x ** 4) / 768 + (625 * x ** 3) / 192 + (125 * x ** 2) / 128 - (125 * x) / 192 - 3 / 256,
                      (15625 * x ** 4) / 768 - (625 * x ** 3) / 64 - (1625 * x ** 2) / 128 + (325 * x) / 64 + 125 / 768,
                      - (15625 * x ** 4) / 384 + (625 * x ** 3) / 96 + (2125 * x ** 2) / 64 - (
                              425 * x) / 96 - 375 / 128,
                      (15625 * x ** 4) / 384 + (625 * x ** 3) / 96 - (2125 * x ** 2) / 64 - (425 * x) / 96 + 375 / 128,
                      - (15625 * x ** 4) / 768 - (625 * x ** 3) / 64 + (1625 * x ** 2) / 128 + (
                              325 * x) / 64 - 125 / 768,
                      (3125 * x ** 4) / 768 + (625 * x ** 3) / 192 - (125 * x ** 2) / 128 - (125 * x) / 192 + 3 / 256])
    elif n == 7:
        B = np.array(
            [(243 * x ** 5) / 40 - (81 * x ** 4) / 16 - (9 * x ** 3) / 4 + (27 * x ** 2) / 16 + x / 10 - 1 / 20,
             - (729 * x ** 5) / 20 + (81 * x ** 4) / 4 + 27 * x ** 3 - (27 * x ** 2) / 2 - (27 * x) / 20 + 9 / 20,
             (729 * x ** 5) / 8 - (405 * x ** 4) / 16 - (351 * x ** 3) / 4 + (351 * x ** 2) / 16 + (27 * x) / 2 - 9 / 4,
             -(x * (243 * x ** 4 - 252 * x ** 2 + 49)) / 2,
             (729 * x ** 5) / 8 + (405 * x ** 4) / 16 - (351 * x ** 3) / 4 - (351 * x ** 2) / 16 + (27 * x) / 2 + 9 / 4,
             - (729 * x ** 5) / 20 - (81 * x ** 4) / 4 + 27 * x ** 3 + (27 * x ** 2) / 2 - (27 * x) / 20 - 9 / 20,
             (243 * x ** 5) / 40 + (81 * x ** 4) / 16 - (9 * x ** 3) / 4 - (27 * x ** 2) / 16 + x / 10 + 1 / 20])
    elif n == 8:
        B = np.array([- (823543 * x ** 6) / 92160 + (117649 * x ** 5) / 15360 + (84035 * x ** 4) / 18432 - (
                16807 * x ** 3) / 4608 - (12691 * x ** 2) / 30720 + (12691 * x) / 46080 + 5 / 2048,
                      (5764801 * x ** 6) / 92160 - (117649 * x ** 5) / 3072 - (991613 * x ** 4) / 18432 + (
                              141659 * x ** 3) / 4608 + (171157 * x ** 2) / 30720 - (
                              24451 * x) / 9216 - 343 / 10240,
                      - (5764801 * x ** 6) / 30720 + (352947 * x ** 5) / 5120 + (420175 * x ** 4) / 2048 - (
                              36015 * x ** 3) / 512 - (445557 * x ** 2) / 10240 + (63651 * x) / 5120 + 1715 / 6144,
                      (5764801 * x ** 6) / 18432 - (117649 * x ** 5) / 3072 - (6974905 * x ** 4) / 18432 + (
                              199283 * x ** 3) / 4608 + (648613 * x ** 2) / 6144 - (92659 * x) / 9216 - 8575 / 2048,
                      - (5764801 * x ** 6) / 18432 - (117649 * x ** 5) / 3072 + (6974905 * x ** 4) / 18432 + (
                              199283 * x ** 3) / 4608 - (648613 * x ** 2) / 6144 - (92659 * x) / 9216 + 8575 / 2048,
                      (5764801 * x ** 6) / 30720 + (352947 * x ** 5) / 5120 - (420175 * x ** 4) / 2048 - (
                              36015 * x ** 3) / 512 + (445557 * x ** 2) / 10240 + (63651 * x) / 5120 - 1715 / 6144,
                      - (5764801 * x ** 6) / 92160 - (117649 * x ** 5) / 3072 + (991613 * x ** 4) / 18432 + (
                              141659 * x ** 3) / 4608 - (171157 * x ** 2) / 30720 - (
                              24451 * x) / 9216 + 343 / 10240,
                      (823543 * x ** 6) / 92160 + (117649 * x ** 5) / 15360 - (84035 * x ** 4) / 18432 - (
                              16807 * x ** 3) / 4608 + (12691 * x ** 2) / 30720 + (12691 * x) / 46080 - 5 / 2048])
    elif n == 9:
        B = np.array([(4096 * x ** 7) / 315 - (512 * x ** 6) / 45 - (128 * x ** 5) / 15 + (64 * x ** 4) / 9 + (
                56 * x ** 3) / 45 - (14 * x ** 2) / 15 - x / 35 + 1 / 70,
                      - (32768 * x ** 7) / 315 + (1024 * x ** 6) / 15 + (512 * x ** 5) / 5 - 64 * x ** 4 - (
                              256 * x ** 3) / 15 + (48 * x ** 2) / 5 + (128 * x) / 315 - 16 / 105,
                      (16384 * x ** 7) / 45 - (7168 * x ** 6) / 45 - (6656 * x ** 5) / 15 + (1664 * x ** 4) / 9 + (
                              5408 * x ** 3) / 45 - (676 * x ** 2) / 15 - (16 * x) / 5 + 4 / 5,
                      - (32768 * x ** 7) / 45 + (7168 * x ** 6) / 45 + (14848 * x ** 5) / 15 - (1856 * x ** 4) / 9 - (
                              15616 * x ** 3) / 45 + (976 * x ** 2) / 15 + (128 * x) / 5 - 16 / 5,
                      (2 * x * (4096 * x ** 6 - 5760 * x ** 4 + 2184 * x ** 2 - 205)) / 9,
                      - (32768 * x ** 7) / 45 - (7168 * x ** 6) / 45 + (14848 * x ** 5) / 15 + (1856 * x ** 4) / 9 - (
                              15616 * x ** 3) / 45 - (976 * x ** 2) / 15 + (128 * x) / 5 + 16 / 5,
                      (16384 * x ** 7) / 45 + (7168 * x ** 6) / 45 - (6656 * x ** 5) / 15 - (1664 * x ** 4) / 9 + (
                              5408 * x ** 3) / 45 + (676 * x ** 2) / 15 - (16 * x) / 5 - 4 / 5,
                      - (32768 * x ** 7) / 315 - (1024 * x ** 6) / 15 + (512 * x ** 5) / 5 + 64 * x ** 4 - (
                              256 * x ** 3) / 15 - (48 * x ** 2) / 5 + (128 * x) / 315 + 16 / 105,
                      (4096 * x ** 7) / 315 + (512 * x ** 6) / 45 - (128 * x ** 5) / 15 - (64 * x ** 4) / 9 + (
                              56 * x ** 3) / 45 + (14 * x ** 2) / 15 - x / 35 - 1 / 70])
    elif n == 10:
        B = np.array([- (43046721 * x ** 8) / 2293760 + (4782969 * x ** 7) / 286720 + (1240029 * x ** 6) / 81920 - (
                531441 * x ** 5) / 40960 - (102789 * x ** 4) / 32768 + (102789 * x ** 3) / 40960 + (
                              87183 * x ** 2) / 573440 - (29061 * x) / 286720 - 35 / 65536,
                      (387420489 * x ** 8) / 2293760 - (4782969 * x ** 7) / 40960 - (15411789 * x ** 6) / 81920 + (
                              5137263 * x ** 5) / 40960 + (1449981 * x ** 4) / 32768 - (
                              1127763 * x ** 3) / 40960 - (1288143 * x ** 2) / 573440 + (
                              47709 * x) / 40960 + 3645 / 458752,
                      - (387420489 * x ** 8) / 573440 + (4782969 * x ** 7) / 14336 + (3720087 * x ** 6) / 4096 - (
                              885735 * x ** 5) / 2048 - (2473497 * x ** 4) / 8192 + (274833 * x ** 3) / 2048 + (
                              496449 * x ** 2) / 28672 - (91935 * x) / 14336 - 5103 / 81920,
                      (129140163 * x ** 8) / 81920 - (4782969 * x ** 7) / 10240 - (48361131 * x ** 6) / 20480 + (
                              6908733 * x ** 5) / 10240 + (8063469 * x ** 4) / 8192 - (2687823 * x ** 3) / 10240 - (
                              2155491 * x ** 2) / 20480 + (239499 * x) / 10240 + 6615 / 16384,
                      - (387420489 * x ** 8) / 163840 + (4782969 * x ** 7) / 20480 + (152523567 * x ** 6) / 40960 - (
                              7263027 * x ** 5) / 20480 - (28258227 * x ** 4) / 16384 + (
                              3139803 * x ** 3) / 20480 + (9974907 * x ** 2) / 40960 - (
                              369441 * x) / 20480 - 178605 / 32768,
                      (387420489 * x ** 8) / 163840 + (4782969 * x ** 7) / 20480 - (152523567 * x ** 6) / 40960 - (
                              7263027 * x ** 5) / 20480 + (28258227 * x ** 4) / 16384 + (
                              3139803 * x ** 3) / 20480 - (9974907 * x ** 2) / 40960 - (
                              369441 * x) / 20480 + 178605 / 32768,
                      - (129140163 * x ** 8) / 81920 - (4782969 * x ** 7) / 10240 + (48361131 * x ** 6) / 20480 + (
                              6908733 * x ** 5) / 10240 - (8063469 * x ** 4) / 8192 - (2687823 * x ** 3) / 10240 + (
                              2155491 * x ** 2) / 20480 + (239499 * x) / 10240 - 6615 / 16384,
                      (387420489 * x ** 8) / 573440 + (4782969 * x ** 7) / 14336 - (3720087 * x ** 6) / 4096 - (
                              885735 * x ** 5) / 2048 + (2473497 * x ** 4) / 8192 + (274833 * x ** 3) / 2048 - (
                              496449 * x ** 2) / 28672 - (91935 * x) / 14336 + 5103 / 81920,
                      - (387420489 * x ** 8) / 2293760 - (4782969 * x ** 7) / 40960 + (15411789 * x ** 6) / 81920 + (
                              5137263 * x ** 5) / 40960 - (1449981 * x ** 4) / 32768 - (
                              1127763 * x ** 3) / 40960 + (1288143 * x ** 2) / 573440 + (
                              47709 * x) / 40960 - 3645 / 458752,
                      (43046721 * x ** 8) / 2293760 + (4782969 * x ** 7) / 286720 - (1240029 * x ** 6) / 81920 - (
                              531441 * x ** 5) / 40960 + (102789 * x ** 4) / 32768 + (102789 * x ** 3) / 40960 - (
                              87183 * x ** 2) / 573440 - (29061 * x) / 286720 + 35 / 65536])
    elif n == 11:
        B = np.array([(1953125 * x ** 9) / 72576 - (390625 * x ** 8) / 16128 - (78125 * x ** 7) / 3024 + (
                78125 * x ** 6) / 3456 + (8125 * x ** 5) / 1152 - (40625 * x ** 4) / 6912 - (
                              5125 * x ** 3) / 9072 + (5125 * x ** 2) / 12096 + x / 126 - 1 / 252,
                      - (9765625 * x ** 9) / 36288 + (390625 * x ** 8) / 2016 + (1015625 * x ** 7) / 3024 - (
                              203125 * x ** 6) / 864 - (59375 * x ** 5) / 576 + (59375 * x ** 4) / 864 + (
                              157625 * x ** 3) / 18144 - (31525 * x ** 2) / 6048 - (125 * x) / 1008 + 25 / 504,
                      (9765625 * x ** 9) / 8064 - (1171875 * x ** 8) / 1792 - (1796875 * x ** 7) / 1008 + (
                              359375 * x ** 6) / 384 + (90625 * x ** 5) / 128 - (90625 * x ** 4) / 256 - (
                              67625 * x ** 3) / 1008 + (13525 * x ** 2) / 448 + (125 * x) / 126 - 25 / 84,
                      - (9765625 * x ** 9) / 3024 + (390625 * x ** 8) / 336 + (1328125 * x ** 7) / 252 - (
                              265625 * x ** 6) / 144 - (40625 * x ** 5) / 16 + (40625 * x ** 4) / 48 + (
                              546125 * x ** 3) / 1512 - (109225 * x ** 2) / 1008 - (125 * x) / 21 + 25 / 21,
                      (9765625 * x ** 9) / 1728 - (390625 * x ** 8) / 384 - (78125 * x ** 7) / 8 + (
                              109375 * x ** 6) / 64 + (1009375 * x ** 5) / 192 - (1009375 * x ** 4) / 1152 - (
                              208625 * x ** 3) / 216 + (41725 * x ** 2) / 288 + (125 * x) / 3 - 25 / 6,
                      -(x * (1953125 * x ** 8 - 3437500 * x ** 6 + 1918125 * x ** 4 - 382250 * x ** 2 + 21076)) / 288,
                      (9765625 * x ** 9) / 1728 + (390625 * x ** 8) / 384 - (78125 * x ** 7) / 8 - (
                              109375 * x ** 6) / 64 + (1009375 * x ** 5) / 192 + (1009375 * x ** 4) / 1152 - (
                              208625 * x ** 3) / 216 - (41725 * x ** 2) / 288 + (125 * x) / 3 + 25 / 6,
                      - (9765625 * x ** 9) / 3024 - (390625 * x ** 8) / 336 + (1328125 * x ** 7) / 252 + (
                              265625 * x ** 6) / 144 - (40625 * x ** 5) / 16 - (40625 * x ** 4) / 48 + (
                              546125 * x ** 3) / 1512 + (109225 * x ** 2) / 1008 - (125 * x) / 21 - 25 / 21,
                      (9765625 * x ** 9) / 8064 + (1171875 * x ** 8) / 1792 - (1796875 * x ** 7) / 1008 - (
                              359375 * x ** 6) / 384 + (90625 * x ** 5) / 128 + (90625 * x ** 4) / 256 - (
                              67625 * x ** 3) / 1008 - (13525 * x ** 2) / 448 + (125 * x) / 126 + 25 / 84,
                      - (9765625 * x ** 9) / 36288 - (390625 * x ** 8) / 2016 + (1015625 * x ** 7) / 3024 + (
                              203125 * x ** 6) / 864 - (59375 * x ** 5) / 576 - (59375 * x ** 4) / 864 + (
                              157625 * x ** 3) / 18144 + (31525 * x ** 2) / 6048 - (125 * x) / 1008 - 25 / 504,
                      (1953125 * x ** 9) / 72576 + (390625 * x ** 8) / 16128 - (78125 * x ** 7) / 3024 - (
                              78125 * x ** 6) / 3456 + (8125 * x ** 5) / 1152 + (40625 * x ** 4) / 6912 - (
                              5125 * x ** 3) / 9072 - (5125 * x ** 2) / 12096 + x / 126 + 1 / 252])
    else:
        B = np.empty(())
    return B


def CFEM_length(n: int) -> np.ndarray:
    """
    Compute the lengths of complex finite elements

    Parameters:
        n (int): The number of elements.

    Returns:
        np.ndarray: Array of lengths of complex finite elements.
    """
    if n == 0:
        arr = np.array([])
    elif n == 1:
        arr = np.array([1. + 0.j])
    elif n == 2:
        arr = np.array([0.4999999999999999 + 0.2886751345948128j,
                        0.4999999999999999 - 0.2886751345948128j])
    elif n == 3:
        arr = np.array([0.2846855768838882 + 0.2715998514163075j,
                        0.4306288462322235 + 0.j,
                        0.2846855768838882 - 0.2715998514163075j])
    elif n == 4:
        arr = np.array([0.1831324805314353 + 0.2313252260262552j,
                        0.3168675194685647 - 0.0948820251422178j,
                        0.3168675194685647 + 0.0948820251422178j,
                        0.1831324805314353 - 0.2313252260262552j])
    elif n == 5:
        arr = np.array([0.1280366783154106 + 0.1966821383462182j,
                        0.2348545087193966 - 0.1220994076370763j,
                        0.2742176259303863 + 0.j,
                        0.2348545087193966 + 0.1220994076370763j,
                        0.1280366783154106 - 0.1966821383462182j])
    elif n == 6:
        arr = np.array([0.0948906178960753 + 0.1694451481943345j,
                        0.1791464073974963 - 0.1259432494634055j,
                        0.2259629747064281 - 0.0461413567178009j,
                        0.2259629747064281 + 0.0461413567178009j,
                        0.1791464073974963 + 0.1259432494634055j,
                        0.0948906178960753 - 0.1694451481943345j])
    elif n == 7:
        arr = np.array([0.0733855956863625 + 0.148119407414618j,
                        0.1406573939584806 - 0.1215478183323456j,
                        0.1853895455326701 - 0.0677649778878383j,
                        0.2011349296449717 + 0.j,
                        0.1853895455326701 + 0.0677649778878383j,
                        0.1406573939584806 + 0.1215478183323456j,
                        0.0733855956863625 - 0.148119407414618j])
    elif n == 8:
        arr = np.array([0.0586179149223442 + 0.1311923697456496j,
                        0.1132583300497219 - 0.1144549641390556j,
                        0.1533779477188381 - 0.0770943035389559j,
                        0.1747458073090958 + 0.0271322617377081j,
                        0.1747458073090958 - 0.0271322617377081j,
                        0.1533779477188381 + 0.0770943035389559j,
                        0.1132583300497219 + 0.1144549641390556j,
                        0.0586179149223442 - 0.1311923697456496j])
    elif n == 9:
        arr = np.array([0.0480204990788967 + 0.1175257048808047j,
                        0.0931628717397013 - 0.1067971799036964j,
                        0.1284044105291807 - 0.08014721079993j,
                        0.1510095702608015 + 0.042815465096303j,
                        0.1588052967828396 + 0.j,
                        0.1510095702608015 - 0.042815465096303j,
                        0.1284044105291807 + 0.08014721079993j,
                        0.0931628717397013 + 0.1067971799036964j,
                        0.0480204990788967 - 0.1175257048808047j])
    elif n == 10:
        arr = np.array([0.0401447291006297 + 0.1063069779668691j,
                        0.0780227361653891 - 0.0994000358123099j,
                        0.1088187481692871 - 0.0799601506040509j,
                        0.1307825645357297 + 0.05163001172989j,
                        0.1422312220289637 + 0.0178284138203735j,
                        0.1422312220289637 - 0.0178284138203735j,
                        0.1307825645357297 - 0.05163001172989j,
                        0.1088187481692871 + 0.0799601506040509j,
                        0.0780227361653891 + 0.0994000358123099j,
                        0.0401447291006297 - 0.1063069779668691j])
    elif n == 11:
        arr = np.array([0.0341226165779975 + 0.0969578962629273j,
                        0.0663448638108099 - 0.0925600518223358j,
                        0.0932908802582502 - 0.0781136664568789j,
                        0.1138607246777459 + 0.0562861684427892j,
                        0.1267842593015181 + 0.0294268479937697j,
                        0.1311933107473581 + 0.j,
                        0.1267842593015181 - 0.0294268479937697j,
                        0.1138607246777459 - 0.0562861684427892j,
                        0.0932908802582502 + 0.0781136664568789j,
                        0.0663448638108099 + 0.0925600518223358j,
                        0.0341226165779975 - 0.0969578962629273j])
    elif n == 12:
        arr = np.array([0.0294080394481397 + 0.0890618139566227j,
                        0.0571519245670593 - 0.0863535253011692j,
                        0.0808258207673494 - 0.0754528907185029j,
                        0.0997629074187518 + 0.0583988566315484j,
                        0.1130139581371071 + 0.0368701962403143j,
                        0.1198373496615945 - 0.0125986648726254j,
                        0.1198373496615945 + 0.0125986648726254j,
                        0.1130139581371071 - 0.0368701962403143j,
                        0.0997629074187518 - 0.0583988566315484j,
                        0.0808258207673494 + 0.0754528907185029j,
                        0.0571519245670593 + 0.0863535253011692j,
                        0.0294080394481397 - 0.0890618139566227j])
    elif n == 13:
        arr = np.array([0.0256431877537112 + 0.0823135532108659j,
                        0.049785739463991 - 0.0807658209696758j,
                        0.0706939092597027 - 0.0724383305064331j,
                        0.087992471454901 + 0.0589456300863142j,
                        0.1009746934200988 + 0.0415298668700282j,
                        0.1090297769583542 - 0.0214431421223852j,
                        0.1117604433784812 + 0.j,
                        0.1090297769583542 + 0.0214431421223852j,
                        0.1009746934200988 - 0.0415298668700282j,
                        0.087992471454901 - 0.0589456300863142j,
                        0.0706939092597027 + 0.0724383305064331j,
                        0.049785739463991 + 0.0807658209696758j,
                        0.0256431877537112 - 0.0823135532108659j])
    elif n == 14:
        arr = np.array([0.0225855031164879 + 0.0764856937396583j,
                        0.0437912763121583 - 0.0757474094883687j,
                        0.0623602777963879 - 0.0693230061100848j,
                        0.078115469373319 + 0.0585285392547126j,
                        0.0905317874679985 + 0.0443119512873602j,
                        0.0991146445862634 - 0.0276025520428727j,
                        0.1035010413473852 - 0.0093715318266685j,
                        0.1035010413473852 + 0.0093715318266685j,
                        0.0991146445862634 + 0.0276025520428727j,
                        0.0905317874679985 - 0.0443119512873602j,
                        0.078115469373319 - 0.0585285392547126j,
                        0.0623602777963879 + 0.0693230061100848j,
                        0.0437912763121583 + 0.0757474094883687j,
                        0.0225855031164879 - 0.0764856937396583j])
    elif n == 15:
        arr = np.array([0.0200657073034169 + 0.0714059192039039j,
                        0.0388463896677584 - 0.0712384906532916j,
                        0.0554299116023426 - 0.0662451101038077j,
                        0.0697748750597596 + 0.0575243236957602j,
                        0.0814923130987373 + 0.045819622675074j,
                        0.0901849037031393 - 0.0318375027497833j,
                        0.0955351449368114 - 0.0163098583377296j,
                        0.0973415092560675 + 0.j,
                        0.0955351449368114 + 0.0163098583377296j,
                        0.0901849037031393 + 0.0318375027497833j,
                        0.0814923130987373 - 0.045819622675074j,
                        0.0697748750597596 - 0.0575243236957602j,
                        0.0554299116023426 + 0.0662451101038077j,
                        0.0388463896677584 + 0.0712384906532916j,
                        0.0200657073034169 - 0.0714059192039039j])
    elif n == 16:
        arr = np.array([0.0179626649633344 + 0.0669416236663275j,
                        0.0347180950757361 - 0.0671796447093068j,
                        0.0496079493066401 - 0.0632780348579755j,
                        0.0626839820429751 + 0.0561718854848456j,
                        0.0736595739404183 + 0.0464585844916812j,
                        0.0822166729805567 - 0.0346871787078374j,
                        0.0880837466425991 - 0.0214195799558673j,
                        0.09106731504774 + 0.0072417524903577j,
                        0.09106731504774 - 0.0072417524903577j,
                        0.0880837466425991 + 0.0214195799558673j,
                        0.0822166729805567 + 0.0346871787078374j,
                        0.0736595739404183 - 0.0464585844916812j,
                        0.0626839820429751 - 0.0561718854848456j,
                        0.0496079493066401 + 0.0632780348579755j,
                        0.0347180950757361 + 0.0671796447093068j,
                        0.0179626649633344 - 0.0669416236663275j])
    else:
        approx_flag = 1  # 0:uniform; 1:Pade; 2:Taylor
        order = n
        j = np.arange(order, -1, -1)
        if approx_flag == 0:  # uniform real lengths
            arr = np.ones(order + 1) / order
        elif approx_flag == 1:  # Diagonal Pade expansion
            c = (-1) ** j / sp.special.factorial(j) * sp.special.factorial(2 * order - j) / sp.special.factorial(
                order - j)
            arr = 2 / np.roots(c)
        elif approx_flag == 2:  # Taylor's expansion
            c = (-1) ** j / sp.special.factorial(j)
            arr = 1 / np.roots(c)
        else:
            raise ValueError('error')
        steps = np.column_stack([-np.angle(arr), np.real(arr), np.imag(arr)])
        steps = steps[np.lexsort(steps.T[::-1])]
        arr = steps[:, 1] + 1j * steps[:, 2]
        for order_indx in range(1, order // 2 + 1):
            if order_indx % 4 == 2 or order_indx % 4 == 3:
                arr[order_indx - 1] = np.conj(arr[order_indx - 1])
                arr[order - order_indx] = np.conj(arr[order - order_indx])
    return arr
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import ForwardHS_First, ForwardHS_Eff, add_optimization
from src.Optimization import WaveDispOptim
from src.Utilities import ParametersOpt

import pickle


def visualize_optimization(PrOpt, synthetic_param, optimized_param,
                           init_params, w_vals, cp_observed, cp_optimized, cp_initial):
    plt.clf()
    plt.plot(w_vals, cp_observed, label='Ground truth')
    plt.plot(w_vals, cp_optimized, label='Optimized', linestyle='dotted', linewidth=3)
    plt.plot(w_vals, cp_initial, label='Initial guess')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Velocity (m/s)')
    plt.box(True)
    plt.legend()
    plt.title(f'param: {np.round(optimized_param, decimals=2)}')
    plt.savefig(f"{PrOpt['out_dir']}optim_cp.png")
    with open(f"{PrOpt['out_dir']}optim_cp.pkl", 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.show() if PrOpt['show_plots'] else None

    plt.clf()
    samples, labels = [synthetic_param, optimized_param, init_params], ['Ground truth', 'Optimized', 'Initial guess']
    style = [('-', 1), ('dotted', 3), ('-', 1)]
    velocities, thicknesses = None, None
    for idx, row in enumerate(samples):
        nr = (len(row) - 1) // 2
        velocities = np.array(row[:nr + 1])
        velocities = np.insert(velocities, 0, velocities[0])
        thicknesses = np.append(row[nr + 1:], 10)
        cumulative_thickness = np.cumsum(thicknesses)
        cumulative_thickness = np.insert(cumulative_thickness, 0, 0.)
        plt.step(velocities, -1 * cumulative_thickness, where='post', label=labels[idx],
                 linestyle=style[idx][0], linewidth=style[idx][1])
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.title(f'Velocity Profile_{str(velocities.astype(int))}_{str(thicknesses.astype(int))}')
    plt.legend()
    plt.grid(True)
    # plt.xlim(100, 600)
    # plt.ylim(0, np.max(samples[,:]))
    plt.savefig(f"{PrOpt['out_dir']}optim_profile.png")
    with open(f"{PrOpt['out_dir']}optim_profile.pkl", 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.show() if PrOpt['show_plots'] else None


def optimization_test_h_flexible():
    synthetic_param = np.array([300., 200., 400., 500., 6., 4., 8.])
    init_params = np.array([320., 320., 320., 190., 190., 415., 415., 415., 415., 530., 1.9])
    forward_func = ForwardHS_First
    # forward_func = ForwardHS_Eff
    PrOpt = ParametersOpt
    PrOpt['h_fixed'] = 0.
    PrOpt['verbose'] = True
    PrOpt['out_dir'] = '../out/out_optim'
    # PrOpt['method'] = 'trust-constr'
    # PrOpt['grad'] = 'FDM'  # 'FDM', 2-point, 3-point, cs
    # PrOpt['hess'] = 'FDM'  # 'FDM', 2-point, 3-point, cs
    PrOpt['max_iter'] = 40.
    PrOpt['grad_eps'] = 1.e-2  # relative to each parameter
    PrOpt['early_stop'] = 1.e-6
    PrOpt['bounds'] = .1
    PrOpt['constant_bounds'] = ((100., 500.), (1.5, 2.5))
    w_vals, cp_observed = forward_func(synthetic_param)
    optimized_param = WaveDispOptim(PrOpt=PrOpt, forward_func=forward_func, cp_observed=cp_observed,
                                    init_params=init_params, n_proc=0)
    n = len(init_params)
    optimized_param = np.pad(optimized_param, (0, n - 3), mode='constant', constant_values=optimized_param[-1])
    init_params = np.pad(init_params, (0, n - 3), mode='constant', constant_values=init_params[-1])
    _, cp_optimized = forward_func(optimized_param)
    _, cp_initial = forward_func(init_params)

    visualize_optimization(PrOpt, synthetic_param, optimized_param,
                           init_params, w_vals, cp_observed, cp_optimized, cp_initial)


def optimization_test_h_fixed():
    synthetic_param = np.array([300., 200., 400., 500., 6., 4., 8.])
    init_params = np.array([350., 350., 350., 250., 250., 450., 450., 450., 450., 550.])
    forward_func = ForwardHS_First
    # forward_func = ForwardHS_Eff
    PrOpt = ParametersOpt
    PrOpt['h_fixed'] = 2.
    PrOpt['verbose'] = True
    PrOpt['out_dir'] = '../out/out_optim'
    # PrOpt['method'] = 'trust-constr'
    # PrOpt['grad'] = 'FDM'  # 'FDM', 2-point, 3-point, cs
    # PrOpt['hess'] = 'FDM'  # 'FDM', 2-point, 3-point, cs
    PrOpt['max_iter'] = 40.
    PrOpt['grad_eps'] = 1.e-2  # relative to each parameter
    PrOpt['early_stop'] = 1.e-6
    PrOpt['bounds'] = .1
    PrOpt['constant_bounds'] = ((100., 500.), (1., 10.))
    w_vals, cp_observed = forward_func(synthetic_param)
    optimized_param = WaveDispOptim(PrOpt=PrOpt, forward_func=forward_func, cp_observed=cp_observed,
                                    init_params=init_params, n_proc=0)
    n = len(init_params)
    optimized_param = np.pad(optimized_param, (0, n - 1), mode='constant', constant_values=PrOpt['h_fixed'])
    init_params = np.pad(init_params, (0, n - 1), mode='constant', constant_values=PrOpt['h_fixed'])
    _, cp_optimized = forward_func(optimized_param)
    _, cp_initial = forward_func(init_params)

    visualize_optimization(PrOpt, synthetic_param, optimized_param,
                           init_params, w_vals, cp_observed, cp_optimized, cp_initial)


def optimization_test():
    synthetic_param = np.array([300., 200., 400., 500., 6., 4., 8.])
    init_params = np.array([350., 250., 450., 550., 7., 5., 7.])
    forward_func = ForwardHS_First
    # forward_func = ForwardHS_Eff
    PrOpt = ParametersOpt
    PrOpt['verbose'] = True
    PrOpt['out_dir'] = '../out/out_optim'
    # PrOpt['method'] = 'trust-constr'
    # PrOpt['grad'] = 'FDM'  # 'FDM', 2-point, 3-point, cs
    # PrOpt['hess'] = 'FDM'  # 'FDM', 2-point, 3-point, cs
    # PrOpt['max_iter'] = 40
    # PrOpt['grad_eps'] = 1.e-2  # relative to each parameter
    # PrOpt['early_stop'] = 1.e-6
    # PrOpt['bounds'] = .1
    # PrOpt['constant_bounds'] = ((100., 500.), (1., 10.))
    w_vals, cp_observed = forward_func(synthetic_param)
    optimized_param = WaveDispOptim(PrOpt=PrOpt, forward_func=forward_func, cp_observed=cp_observed,
                                    init_params=init_params, n_proc=0)
    _, cp_optimized = forward_func(optimized_param)
    _, cp_initial = forward_func(init_params)

    visualize_optimization(PrOpt, synthetic_param, optimized_param,
                           init_params, w_vals, cp_observed, cp_optimized, cp_initial)


def add_optimization_test():
    PrOpt = ParametersOpt
    # forward_func = ForwardHS_First
    forward_func = ForwardHS_Eff
    PrOpt['max_iter'] = 5
    PrOpt['h_fixed'] = -1.
    PrOpt['constant_bounds'] = ((150., 550.), (1.5, 8.5))
    add_optimization(PrOpt, forward_func=forward_func,
                     search_dir='../out/out_train', output_dir=f'',
                     num_proc=0, num_proc_grad=0)


def add_multi_optimization_test():
    PrOpt = ParametersOpt
    # forward_func = ForwardHS_First
    forward_func = ForwardHS_Eff
    for i in range(1, 3):  # [1,2]
        PrOpt['max_iter'] = i
        PrOpt['h_fixed'] = -1.
        PrOpt['constant_bounds'] = ((150., 550.), (1.5, 8.5))
        add_optimization(PrOpt, forward_func=forward_func,
                         search_dir='../out/out_train', output_dir=f'../out/out_train/optim_{i}',
                         num_proc=0, num_proc_grad=0)


if __name__ == "__main__":
    optimization_test()
    # optimization_test_h_flexible()
    # optimization_test_h_fixed()
    # add_optimization_test()
    # add_multi_optimization_test()

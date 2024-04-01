import os
import pickle
from time import time
from typing import Tuple
from psutil import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize, OptimizeResult
from concurrent.futures import ThreadPoolExecutor

from src.Utilities import ParametersOpt

len_cp, jac = 0, np.empty([])
param_values, misfit_values, verbose, out_log_path, = [], [], False, ''
iter_count, patience, early_stop_threshold = 0, 0, 0.


class StopOptimizationException(Exception):
    pass


def callback_func(intermediate_result: OptimizeResult) -> None:
    """
    Callback function for optimization iterations.

    Parameters:
    - intermediate_result (OptimizeResult): Intermediate result of the optimization.

    Returns:
    None
    """
    global iter_count, verbose, patience
    iter_count += 1
    misfit_values.append(intermediate_result.fun)
    param_values.append(intermediate_result['x'])
    with open(out_log_path, "a") as f:
        param_str = ' '.join(map(str, np.round(param_values[-1], decimals=12)))
        iter_hist = f"Iter {iter_count} > misfit: {np.round(misfit_values[-1], decimals=12)}, param: [{param_str}]"
        f.write(iter_hist + '\n')
    if verbose and iter_count % 1 == 0:
        param_str = ' '.join(map(str, np.round(param_values[-1], decimals=2)))
        iter_hist = f"Iter {iter_count} > misfit: {np.round(misfit_values[-1], decimals=6)}, param: [{param_str}]"
        print(iter_hist)
    if (early_stop_threshold != 0. and len(misfit_values) > patience and
            all(abs((misfit_values[-i] - misfit_values[-i - 1]) / misfit_values[-i - 1]) < early_stop_threshold
                for i in range(1, patience + 1))):
        msg = f"Early stopping optimization based on threshold: {early_stop_threshold}"
        raise StopOptimizationException(msg)


def forward(m: np.ndarray, forward_func: callable, h_fixed: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward computation function with optional thickness adjustment.

    Parameters:
    - m (np.ndarray): Input array for forward computation.
    - forward_func (callable): Callable function for forward computation.
    - h_fixed (float): Fixed thickness value if applicable.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing phase velocity (cp) and frequency (w).
    """

    n = len(m)
    if h_fixed == 0.:
        m = np.pad(m, (0, n - 3), mode='constant', constant_values=m[-1])
    elif h_fixed == -1.:
        ...  # no change needed
    else:
        m = np.pad(m, (0, n - 1), mode='constant', constant_values=h_fixed)
    w, cp = forward_func(m)
    return cp, w


def calculate_jac_mlp(i: int, m: np.ndarray, epsilon: float, forward_func: callable,
                      per: np.ndarray, h_fixed: float) -> np.ndarray:
    """
    Calculates the Jacobian matrix for a Multi-Layer Perceptron (MLP).

    Parameters:
    - i (int): Index of the parameter.
    - m (np.ndarray): Input array of parameters.
    - epsilon (float): Perturbation factor for finite difference calculation.
    - forward_func (callable): Callable function for forward computation.
    - per (np.ndarray): Array of phase velocity values.
    - h_fixed (float): Fixed thickness value if applicable.

    Returns:
    np.ndarray: column of a Jacobian matrix.
    """

    delta = epsilon * m[i]
    m_plus = m.copy()
    m_plus[i] += delta
    per_plus, _ = forward(m_plus, forward_func, h_fixed)
    return (per_plus - per) / delta


def objective_gradient_mlp(m: np.ndarray, c_observed: np.ndarray, forward_func: callable, p: int,
                           epsilon: float, h_fixed: float) -> Tuple[float, np.ndarray]:
    """
    Calculates the objective function and gradient for a Multi-Layer Perceptron (MLP).

    Parameters:
    - m (np.ndarray): Input array of parameters.
    - c_observed (np.ndarray): Observed phase velocity values.
    - forward_func (callable): Callable function for forward computation.
    - p (int): Number of processes for parallel computation.
    - epsilon (float): Perturbation factor for finite difference calculation.
    - h_fixed (float): Fixed thickness value if applicable.

    Returns:
    Tuple[float, np.ndarray]: Tuple containing the mean squared error (mse) and the gradient array.
    """

    global jac, len_cp
    len_cp = len(c_observed)
    per, _ = forward(m, forward_func, h_fixed)
    diff = (per - c_observed)
    mse = np.sum(diff ** 2) / len(c_observed)
    jac = np.zeros((len(c_observed), len(m)))
    if p == 1:
        for i in range(len(m)):
            jac[:, i] = calculate_jac_mlp(i, m, epsilon, forward_func, per, h_fixed)
    else:
        with Pool(p) as pool:
            jac_values = pool.starmap(calculate_jac_mlp, [(i, m, epsilon, forward_func, per) for i in range(len(m))])
        jac[:, :] = np.array(jac_values).T
    grad = np.dot(jac.T, diff) * (2. / len(c_observed))
    return mse, grad


def objective_gradient(m: np.ndarray, c_observed: np.ndarray, forward_func: callable, p: int,
                       epsilon: float, h_fixed: float) -> Tuple[float, np.ndarray]:
    """
    Calculates the objective function and gradient.

    Parameters:
    - m (np.ndarray): Input array of parameters.
    - c_observed (np.ndarray): Observed phase velocity values.
    - forward_func (callable): Callable function for forward computation.
    - p (int): Number of processes for parallel computation.
    - epsilon (float): Perturbation factor for finite difference calculation.
    - h_fixed (float): Fixed thickness value if applicable.

    Returns:
    Tuple[float, np.ndarray]: Tuple containing the mean squared error (mse) and the gradient array.
    """
    global jac, len_cp
    len_cp = len(c_observed)
    per, _ = forward(m, forward_func, h_fixed)
    diff = (per - c_observed)
    mse = np.sum(diff ** 2) / len(c_observed)
    jac = np.zeros((len(c_observed), len(m)))

    def compute_jac(j):
        delta = epsilon * m[j]
        m_plus = m.copy()
        m_plus[j] += delta
        per_plus, _ = forward(m_plus, forward_func, h_fixed)
        return (per_plus - per) / delta

    if p == 1:
        for i in range(len(m)):
            jac[:, i] = compute_jac(i)
    else:
        with ThreadPoolExecutor(max_workers=p) as executor:
            futures = [executor.submit(compute_jac, i) for i in range(len(m))]
            for i, future in enumerate(futures):
                jac[:, i] = future.result()
    grad = np.dot(jac.T, diff) * (2. / len(c_observed))
    # add regularization term (experimental) ----------------------------------------------------------
    # grad_regularized, regularization_term, regularization_coeff = np.zeros_like(m), 0, 10.
    # bounds = [(100., 700.), (100., 700.), (100., 700.), (100., 700.), (1., 10.), (1., 10.), (1., 10.)]
    # for i, (lower_bound, upper_bound) in enumerate(bounds):
    #     if lower_bound > m[i]:
    #         regularization_term += (lower_bound - m[i]) ** 2
    #         grad_regularized[i] += -2 * (lower_bound - m[i])
    #     elif upper_bound < m[i]:
    #         regularization_term += (m[i] - upper_bound) ** 2
    #         grad_regularized[i] += 2 * (m[i] - upper_bound)
    # if regularization_term > 0.:
    #     mse = mse + regularization_term * regularization_coeff
    #     grad = grad + grad_regularized * regularization_coeff
    # -----------------------------------------------------------------------------------------------------

    return mse, grad


def hessian(m: np.ndarray, *args) -> np.ndarray:
    """
    Calculates the Hessian matrix.

    Parameters:
    - m (np.ndarray): Input array of parameters.
    - *args: Additional arguments (not used in this function).

    Returns:
    np.ndarray: Hessian matrix.
    """
    global jac, len_cp
    hess = np.dot(jac.T, jac) * (2. / len_cp)
    return hess


def objective(m: np.ndarray, c_observed: np.ndarray, forward_func: callable, h_fixed: float) -> float:
    """
    Calculates the objective function.

    Parameters:
    - m (np.ndarray): Input array of parameters.
    - c_observed (np.ndarray): Observed phase velocity values.
    - forward_func (callable): Callable function for forward computation.
    - h_fixed (float): Fixed thickness value if applicable.

    Returns:
    float: Mean squared error (mse).
    """
    per, _ = forward(m, forward_func, h_fixed)
    diff = (per - c_observed)
    mse = np.sum(diff ** 2) / len(c_observed)
    return mse


def WaveDispOptim(PrOpt: ParametersOpt, forward_func: callable, cp_observed: np.ndarray, init_params: np.ndarray,
                  n_proc: int = 1) -> np.ndarray:
    """
    Performs wave dispersion optimization.

    Parameters:
    - PrOpt (ParametersOpt): Optimization parameters.
    - forward_func (callable): Callable function for forward computation.
    - cp_observed (np.ndarray): Observed phase velocity values.
    - init_params (np.ndarray): Initial parameters for optimization.
    - n_proc (int, optional): Number of processes for parallel computation (default is 1).

    Returns:
    np.ndarray: Optimized parameters.
    """
    global verbose, out_log_path, early_stop_threshold, len_cp, jac, misfit_values, param_values, iter_count, patience
    early_stop = False
    PrOpt['out_dir'] = PrOpt['out_dir'] + '/' if PrOpt['out_dir'] and not PrOpt['out_dir'].endswith('/') else PrOpt[
        'out_dir']
    if PrOpt['out_dir'] and not os.path.exists(PrOpt['out_dir']):
        os.makedirs(PrOpt['out_dir'])
    verbose, out_log_path = PrOpt['verbose'], f"{PrOpt['out_dir']}optim_{PrOpt['optim_id']}.txt"
    early_stop_threshold, patience = PrOpt['early_stop'], PrOpt['patience']
    len_cp, jac, misfit_values, param_values, iter_count = len(cp_observed), np.empty([]), [], [], 0
    n, h_fixed = len(init_params), PrOpt['h_fixed']
    with open(out_log_path, 'w') as file:
        pass
    if PrOpt['max_iter'] < 1:
        return init_params
    bounds, bound_methods = None, ('Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA')
    if PrOpt['bounds'] != 0. and PrOpt['method'] in bound_methods:
        bounds = [(init_params[i] - PrOpt['bounds'] * init_params[i],
                   init_params[i] + PrOpt['bounds'] * init_params[i]) for i in range(len(init_params))]
    if PrOpt['constant_bounds'] is not None and PrOpt['method'] in bound_methods:
        if PrOpt['h_fixed'] == -1.:
            bounds = ([PrOpt['constant_bounds'][0] for _ in range(n // 2 + 1)] +
                      [PrOpt['constant_bounds'][1] for _ in range(n // 2)])
        elif PrOpt['h_fixed'] == 0.:
            bounds = ([PrOpt['constant_bounds'][0] for _ in range(n - 1)] +
                      [PrOpt['constant_bounds'][1] for _ in range(1)])
        else:
            bounds = [PrOpt['constant_bounds'][0] for _ in range(n)]
    p = cpu_count(logical=True)  # num logical processes
    p = len(init_params) if 0 < len(init_params) < p else p
    p = n_proc if 0 < n_proc < p else p
    epsilon = PrOpt['grad_eps']
    optimized_param = init_params
    t = time()
    try:
        if PrOpt['method'] in ('Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr'):
            if PrOpt['grad'] == PrOpt['hess'] == 'FDM':
                result = minimize(objective_gradient, init_params,
                                  args=(cp_observed, forward_func, p, epsilon, h_fixed),
                                  method=PrOpt['method'], jac=True, hess=hessian, bounds=bounds,
                                  options={'maxiter': PrOpt['max_iter'], 'disp': False}, callback=callback_func)
            elif PrOpt['grad'] != 'FDM' and PrOpt['hess'] != 'FDM':
                result = minimize(objective, init_params, args=(cp_observed, forward_func, h_fixed),
                                  method=PrOpt['method'], jac=PrOpt['grad'], hess=PrOpt['hess'], bounds=bounds,
                                  options={'maxiter': PrOpt['max_iter'], 'disp': False}, callback=callback_func)
            else:
                raise ValueError('Wrong gradient/hessian method')
        elif PrOpt['method'] in ('CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP'):
            if PrOpt['grad'] == 'FDM':
                result = minimize(objective_gradient, init_params,
                                  args=(cp_observed, forward_func, p, epsilon, h_fixed),
                                  method=PrOpt['method'], jac=True, bounds=bounds,
                                  options={'maxiter': PrOpt['max_iter'], 'disp': False}, callback=callback_func)
            else:
                result = minimize(objective, init_params, args=(cp_observed, forward_func, h_fixed),
                                  method=PrOpt['method'], jac=PrOpt['grad'], bounds=bounds,
                                  options={'maxiter': PrOpt['max_iter'], 'disp': False}, callback=callback_func)
        elif PrOpt['method'] in ('Nelder-Mead', 'Powell'):
            result = minimize(objective, init_params, args=(cp_observed, forward_func, h_fixed),
                              method=PrOpt['method'], bounds=bounds,
                              options={'maxiter': PrOpt['max_iter'], 'disp': False}, callback=callback_func)
        else:
            raise ValueError('Wrong method type.')
    except StopOptimizationException as e:
        early_stop, result = True, None
    except Exception as e:
        with open(out_log_path, "a") as f:
            f.write(f"Problem with minimize: {str(e)}, returning initial guess as final param")
        print(f"Problem with minimize: {str(e)}, returning initial guess as final param") if PrOpt['verbose'] else None
    else:
        optimized_param = result.x
    if early_stop is True:
        if len(param_values) > 0:
            optimized_param = param_values[-1]
            msg = f"Early stopping optimization based on threshold: {early_stop_threshold} and patience: {patience}"
            with open(out_log_path, "a") as f:
                f.write('\n' + msg)
            print('\n' + msg) if verbose else None
        else:
            with open(out_log_path, "a") as f:
                f.write("Problem with history array, returning initial guess as final param")
            print("Problem with history array, returning initial guess as final param") if PrOpt['verbose'] else None
    param_str = ' '.join(map(str, np.round(np.array(optimized_param), decimals=3)))
    print(f"\nFinal param: [{param_str}]") if PrOpt['verbose'] else None
    print(f"Optimization time: {time() - t:.4f} seconds") if PrOpt['verbose'] else None
    with open(out_log_path, "a") as f:
        param_str = ' '.join(map(str, np.round(np.array(optimized_param), decimals=12)))
        f.write(f"\nFinal param: [{param_str}]\n")
        f.write(f"Optimization time: {time() - t:.4f} seconds\n")

    # Plot the misfit values
    if len(misfit_values) > 0:
        plt.clf()
        plt.plot(range(1, len(misfit_values) + 1), np.log10(misfit_values))
        plt.xlabel('Iterations')
        plt.ylabel('log10(Misfit)')
        plt.title('Iteration History')
        plt.box(True)
        if PrOpt['save_plots']:
            plt.savefig(f"{PrOpt['out_dir']}optim_hist_{PrOpt['optim_id']}.png")
            with open(f"{PrOpt['out_dir']}optim_hist_{PrOpt['optim_id']}.pkl", 'wb') as f:
                pickle.dump(plt.gcf(), f)
        plt.show() if PrOpt['show_plots'] else None

    return optimized_param

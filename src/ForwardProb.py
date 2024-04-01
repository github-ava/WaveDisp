import itertools
import os
import pickle
import shutil
import traceback
import warnings
from functools import partial
from multiprocessing import Pool
from time import time, sleep
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from psutil import cpu_count

from src.Utilities import Parameters, ParametersNN, ParametersOpt
from src.WaveDisp import WaveDisp
from src.Optimization import WaveDispOptim


def initialize_params(params: np.ndarray) -> Parameters:
    """
    Initializes parameters for the finite element method (FEM) or complex finite element method (CFEM).

    Parameters:
    - params (np.ndarray): Array of parameters including layer shear wave velocity, bottom half-space shear wave velocity,
                            layer thickness, and number of elements per layer.

    Returns:
    - Parameters: An instance of the Parameters class with initialized parameters.
    """
    Pr = Parameters()
    nu = 0.35
    rho = 1800
    n = (len(params) - 1) // 2
    cs = params[:n]
    csB = params[n]
    cp_cs_fac = np.sqrt((2 - 2 * nu) / (1 - 2 * nu))
    h = params[n + 1:]

    # ------------- for FEM-quartic
    Pr.fem = 'fem'  # Method: fem or cfem (Complex-FEM)
    num_el = np.ones_like(cs, dtype=int)
    order = np.ceil(h * 1.2).astype(int)
    order[order < 4] = 4
    order[order > 10] = 10
    # ------------- for CFEM
    # Pr.fem = 'cfem'  # Method: fem or cfem (Complex-FEM)
    # num_el = np.ceil(h / 2).astype(int)
    # num_el[num_el < 4] = 4
    # num_el[num_el > 10] = 10
    # order = np.ones_like(cs, dtype=int)
    # -----------------------

    # SOLID
    Pr.cs = cs  # Layer Shear Wave Velocity (m/s)
    Pr.cp = Pr.cs * cp_cs_fac  # Layer Pressure Wave Velocity (m/s)
    Pr.roS = np.ones_like(Pr.cs) * rho  # Layer Density (kg/m3)
    Pr.h = h  # Layer Thickness (m)
    Pr.nDivS = num_el  # Number of Elements per Layer
    Pr.order = order  # FEM order 1, 2, 3, ..., 10

    # BOTTOM HALF-SPACE [using PMDLs]
    Pr.csB = csB  # HS Shear Wave Velocity (m/s)
    Pr.cpB = csB * cp_cs_fac  # HS Pressure Wave Velocity (m/s)
    Pr.roB = rho  # HS Density (kg/m3)

    # PRINT
    Pr.print = 'no'  # Print info: 'yes','no'
    return Pr


def ForwardHS_First(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward modeling for the first mode of the half-space using the specified parameters.

    Parameters:
    - params (np.ndarray): Array of parameters including layer shear wave velocity, bottom half-space shear wave velocity,
                            layer thickness, and number of elements per layer.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the frequency array (w) and the corresponding phase velocity array (cpT).
    """
    Pr = initialize_params(params)
    Pr.eff = 'no'
    Pr.cg = 'no'
    Pr.nMode = 1
    noise_percentage = 0.  # 5 means 5% Gaussian noise
    warnings.simplefilter("error", RuntimeWarning)
    cp_min = 150.
    cp_max = 550.
    try:
        kz, cg, cpE, cgT, cpT, kzT = WaveDisp(Pr)
        if np.any(np.isnan(cpT[0])) or np.any(np.isinf(cpT[0])):
            cpT[0] = np.zeros_like(Pr.w)
        elif np.min(cpT[0]) < cp_min or np.max(cpT[0]) > cp_max:
            cpT[0] = np.zeros_like(Pr.w)
        elif noise_percentage != 0.:
            # cpT[0] = cpT[0] + (noise_percentage / 100.) * np.random.normal(0, cpT[0].std(), cpT[
            #     0].shape)  # without randomizing the noise for each element
            cpT[0] = cpT[0] + np.random.normal(0, (noise_percentage / 100.) * cpT[0].std(), cpT[0].shape)
    except Exception as e:
        # print('Error in ForwardHS_First: ' + str(e) + '\n' + traceback.format_exc())
        cpT = [np.zeros_like(Pr.w)]
    return Pr.w, cpT[0]


def ForwardHS_Eff(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the effective forward modeling using the specified parameters.

    Parameters:
    - params (np.ndarray): Array of parameters including layer shear wave velocity, bottom half-space shear wave velocity,
                            layer thickness, and number of elements per layer.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the frequency array (w) and the corresponding effective phase velocity array (cpE).
    """
    Pr = initialize_params(params)
    Pr.trace = 'no'
    Pr.cg = 'no'
    noise_percentage = 0.  # 5 means 5% Gaussian noise
    warnings.simplefilter("error", RuntimeWarning)
    cp_min = 150.
    cp_max = 550.
    try:
        kz, cg, cpE, cgT, cpT, kzT = WaveDisp(Pr)
        if np.any(np.isnan(cpE)) or np.any(np.isinf(cpE)) or np.argmax(cpE) != 0:
            cpE = np.zeros_like(Pr.w)
        elif np.min(cpE) < cp_min or np.max(cpE) > cp_max:
            cpE = np.zeros_like(Pr.w)
        elif noise_percentage != 0.:
            # cpE = (cpE + (noise_percentage / 100.) *
            #        np.random.normal(0, cpE.std(), cpE.shape))  # without randomizing the noise for each element
            cpE = cpE + np.random.normal(0, (noise_percentage / 100.) * cpE.std(), cpE.shape)
    except Exception as e:
        # print('Error in ForwardHS_Eff: ' + str(e) + '\n' + traceback.format_exc())
        cpE = np.zeros_like(Pr.w)
    return Pr.w, cpE


def generate_samples(m: int, n: int, cs_min: float, cs_max: float, h_min: float, h_max: float,
                     h_fixed: float = -1., max_scale_factor: float = 3.) -> np.ndarray:
    """
    Generates random samples of layer properties (cs and h) for forward modeling.

    Parameters:
    - m (int): Number of samples to generate.
    - n (int): Number of layers.
    - cs_min (float): Minimum value for layer shear wave velocity (cs).
    - cs_max (float): Maximum value for layer shear wave velocity (cs).
    - h_min (float): Minimum value for layer thickness (h).
    - h_max (float): Maximum value for layer thickness (h).
    - h_fixed (float, optional): Fixed thickness value if not -1. Default is -1. (-1. for variable thickness)
    - max_scale_factor (float, optional): Maximum relative difference allowed for cs or h. Default is 3.

    Returns:
    - np.ndarray: Generated samples, where each row represents a sample with [cs1, cs2, ..., h1, h2, ...] format.
    """
    # h_fixed = -1. means variable thickness for each layer
    #         = 1.  means fixed thickness 1. for all layers
    #         = 0.  means variable thickness but same for all layers
    # max_scale_factor = 3. means maximum 300% relative difference for H or Cs
    samples = np.zeros((m, 2 * n + 1))
    for i in range(m):
        cs_values = np.random.uniform(cs_min, cs_max, n + 1)
        if h_fixed == -1.:
            h_values = np.random.uniform(h_min, h_max, n)
        elif h_fixed == 0.:
            h_values = np.full(n, np.random.uniform(h_min, h_max))
        else:
            h_values = np.random.uniform(h_fixed, h_fixed, n)
        max_index = np.argmax(cs_values)  # Find the index of the maximum value
        cs_values[-1], cs_values[max_index] = cs_values[max_index], cs_values[-1]  # Swap the maximum value to the end
        # scaling -------------------------------
        min_val, max_val = np.min(cs_values), np.max(cs_values)
        if max_val / min_val > max_scale_factor:
            cs_values = (cs_values - min_val) * (
                    (max_val - max_val / max_scale_factor) / (max_val - min_val)) + max_val / max_scale_factor
        min_val, max_val = np.min(h_values), np.max(h_values)
        if max_val / min_val > max_scale_factor:
            h_values = (h_values - min_val) * (
                    (max_val - max_val / max_scale_factor) / (max_val - min_val)) + max_val / max_scale_factor
        # ---------------------------------------
        sample = np.concatenate((cs_values, h_values))
        samples[i] = sample
    indices = np.random.permutation(m)
    samples = samples[indices]
    return samples


def process_params(params: np.ndarray) -> np.ndarray:
    # w, cp = ForwardHS_First(params)
    w, cp = ForwardHS_Eff(params)
    return cp


def parallel_process_samples(samples: np.ndarray, w: np.ndarray, num_proc: int = 1) -> np.ndarray:
    """
    Perform parallel processing on the given samples using multiple processes.

    Parameters:
    - samples (np.ndarray): Array of samples to process.
    - w (np.ndarray): Array of frequencies.
    - num_proc (int, optional): Number of processes to use. Default is 1.

    Returns:
    - np.ndarray: Processed values for each sample.
    """
    p = cpu_count(logical=True)  # num logical processes
    p = len(samples) if 0 < len(samples) < p else p
    p = num_proc if 0 < num_proc < p else p
    with Pool(processes=p) as pool:
        cp_values = pool.map(process_params, samples)
    try:
        cp_values = np.array(cp_values)
    except Exception as e:
        print('Error in parallel_process_samples: ' + str(e))
        # print('Error in parallel_process_samples: ' + str(e) + '\n' + traceback.format_exc())
        max_size = len(w)
        cp_values_padded = []
        for cp in cp_values:
            if len(cp) < max_size:
                cp_padded = np.concatenate((np.full(max_size - len(cp), np.nan), cp))
            else:
                cp_padded = cp
            cp_values_padded.append(cp_padded)
        cp_values = np.array(cp_values_padded)
    return cp_values


def multi_train_network(
        PrOpt: dict,
        PrOptRemoveList: Tuple[dict] = (),
        base_out_dir: str = '',
        cuda_visible: int = -1,
        num_proc: int = 1,
        tf_intra_op: int = 0,
        tf_inter_op: int = 0,
) -> None:
    """
    Perform training of multiple neural network models in parallel based on given optimization parameters.

    Parameters:
    - PrOpt (dict): Dictionary containing optimization parameters.
    - PrOptRemoveList (Tuple[dict], optional): List of dictionaries containing parameters to remove from optimization.
      Default is ().
    - base_out_dir (str, optional): Base output directory path. Default is ''.
    - cuda_visible (int, optional): CUDA visible devices index. Default is -1.
    - num_proc (int, optional): Number of processes to use for parallel training. Default is 1.
    - tf_intra_op (int, optional): TensorFlow intra-op parallelism threads. Default is 0.
    - tf_inter_op (int, optional): TensorFlow inter-op parallelism threads. Default is 0.

    Returns:
    - None
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if not all(isinstance(value, list) for value in PrOpt.values()):
        print("All values in PrOpt should be lists.") if rank == 0 else None
        return
    if not all(key in ParametersNN for key in PrOpt):
        print("All keys in PrOpt should exist in ParametersNN.") if rank == 0 else None
        return
    for PrOptRemove in PrOptRemoveList:
        if len(PrOptRemove) > 0 and not all(isinstance(value, list) for value in PrOptRemove.values()):
            print("All values in PrOptRemove should be lists.") if rank == 0 else None
            return
        if len(PrOptRemove) > 0 and not all(key in ParametersNN for key in PrOptRemove):
            print("All keys in PrOptRemove should exist in ParametersNN.") if rank == 0 else None
            return
    combinations = list(itertools.product(*[PrOpt[key] for key in PrOpt]))

    final_objects = []
    for combo in combinations:
        temp_params = ParametersNN.copy()
        for i, key in enumerate(PrOpt):
            temp_params[key] = combo[i]
        final_objects.append(temp_params)

    def should_remove(dictionary, remove_dict_items):
        for remove_dict_item in remove_dict_items:
            if not remove_dict_item:  # Skip if remove_dict is empty
                continue
            if all(key in dictionary and dictionary[key] in values for key, values in remove_dict_item.items()):
                return True
        return False

    final_objects = [obj for obj in final_objects if not should_remove(obj, PrOptRemoveList)]

    if len(final_objects) == 0:
        print("List of cases to try in PrOpt after removing PrOptRemove cases is empty.") if rank == 0 else None
        return

    base_out_dir = base_out_dir + '/' if base_out_dir and not base_out_dir.endswith('/') else base_out_dir
    sum_dir = f'{base_out_dir}summary/'
    for i, obj in enumerate(final_objects):
        obj['out_dir'] = f'{base_out_dir}out_{i}'
        obj['expr_id'] = i
    parts = np.linspace(0, len(final_objects), size + 1, dtype=int)
    parts = np.diff(parts)
    p_m = cpu_count(logical=False)  # num physical processes
    p = p_m
    p = parts[rank] if 0 < parts[rank] < p else p
    p = num_proc if 0 < num_proc < p else p
    # num_proc = 1 if num_proc == 0 and size == 1 else num_proc
    if cuda_visible < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cuda_visible))
        if tf_intra_op != 0:
            os.environ["TF_NUM_INTRAOP_THREADS"] = str(int(tf_intra_op))
        else:
            if p_m < p:
                print(f"cpu_count < p on rank {rank}.")
                return
            os.environ["TF_NUM_INTRAOP_THREADS"] = str(int(p_m / p))
        if tf_inter_op != 0:
            os.environ["TF_NUM_INTEROP_THREADS"] = str(int(tf_inter_op))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(rank))
    final_obj_list = []
    start_idx = 0
    for size_p in parts:
        end_idx = start_idx + size_p
        final_obj_list.append(final_objects[start_idx:end_idx])
        start_idx = end_idx

    if rank == 0:
        if sum_dir and not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        if base_out_dir and not os.path.exists(base_out_dir):
            os.makedirs(base_out_dir)
        with open(f'{sum_dir}param_list.txt', 'w') as f:
            f.write(f"Common params -----------------------------------\n\n")
            for key, value in PrOpt.items():
                if len(value) == 1:
                    f.write(f"{key}: {value[0]}\n")
            f.write(f"\nExperiment params -------------------------------\n")
            for idx, obj in enumerate(final_objects):
                f.write(f"\n[Experiment {idx}] ----------------------------------\n")
                for key, value in obj.items():
                    if key in PrOpt and len(PrOpt[key]) > 1:
                        f.write(f"{key}: {value}\n")

    comm.Barrier()

    t = time()
    with Pool(processes=p) as pool:
        # result: [cp_error, param_error, train_valid_test_loss]
        results_rank = pool.map(forward_train_network, final_obj_list[rank])
        with open(f"{base_out_dir}results_{rank}.pkl", "wb") as f:
            pickle.dump(results_rank, f)

    comm.Barrier()

    if rank == 0:
        print('Multi training time: {:.3f} s'.format(time() - t))
        results = []
        for i in range(size):
            with open(f"{base_out_dir}results_{i}.pkl", "rb") as f:
                results_rank = pickle.load(f)
                results.extend(results_rank)
        with open(f'{sum_dir}param_list.txt', 'a') as f:
            f.write(f"\nParams and Times -------------------------------\n")
            for idx, obj in enumerate(final_objects):
                f.write(f"\nExperiment params -------------------------------\n")
                model_fit_output_file = os.path.join(obj['out_dir'], 'model_fit_output.txt')
                if os.path.exists(model_fit_output_file):
                    with open(model_fit_output_file, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            if line.startswith('Training time:'):
                                f.write(line)
                for key, value in obj.items():
                    if key in PrOpt and len(PrOpt[key]) > 1:
                        f.write(f"{key}: {value}\n")

        train_loss = np.zeros(len(final_objects))
        valid_loss = np.zeros(len(final_objects))
        test_loss = np.zeros(len(final_objects))

        n_benchmarks = 0
        for i in range(len(final_objects)):
            if np.size(results[i][2]) > 0:
                train_loss[i] = results[i][2][0]
            if np.size(results[i][2]) > 0:
                valid_loss[i] = results[i][2][1]
            if np.size(results[i][2]) > 0:
                test_loss[i] = results[i][2][2]
            n_benchmarks = max(n_benchmarks, len(results[i][0])) if np.size(results[i][0]) > 0 else n_benchmarks

        cp_error = np.zeros((n_benchmarks, len(final_objects)))
        param_error = np.zeros((n_benchmarks, len(final_objects)))

        for i in range(len(final_objects)):
            if np.size(results[i][0]) > 0:
                cp_error[:, i] = results[i][0]
            if np.size(results[i][1]) > 0:
                param_error[:, i] = results[i][1]

        overall_cp_error = np.sum(cp_error, axis=0)
        overall_param_error = np.sum(param_error, axis=0)

        overall_loss = train_loss + valid_loss + test_loss

        np.savetxt(f'{sum_dir}train_loss', train_loss)
        np.savetxt(f'{sum_dir}valid_loss', valid_loss)
        np.savetxt(f'{sum_dir}test_loss', test_loss)
        np.savetxt(f'{sum_dir}cp_error', cp_error)
        np.savetxt(f'{sum_dir}param_error', param_error)

        plt.clf()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(valid_loss, label='Validation Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Experiment')
        plt.ylabel('Loss')
        plt.title(f'train_min:{np.argmin(train_loss)}, valid_min:{np.argmin(valid_loss)}, '
                  f'test_min:{np.argmin(test_loss)}, overall_min:{np.argmin(overall_loss)}')
        plt.legend()
        plt.savefig(f'{sum_dir}train_valid_loss.png')
        with open(f'{sum_dir}train_valid_loss.pkl', 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if final_objects[0]['show_plots'] else None

        plt.clf()
        for i in range(n_benchmarks):
            plt.plot(cp_error[i, :], label=f'cp_error_{i}')
        plt.xlabel('Experiment')
        plt.ylabel('cp_error')
        min_indices = [np.argmin(row) for row in cp_error]
        min_indices_str = ', '.join(map(str, min_indices))
        plt.title('min cp_error: ' + min_indices_str + f' overall: {np.argmin(overall_cp_error)}')
        plt.legend()
        plt.savefig(f'{sum_dir}cp_error.png')
        with open(f'{sum_dir}cp_error.pkl', 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if final_objects[0]['show_plots'] else None

        plt.clf()
        for i in range(n_benchmarks):
            plt.plot(param_error[i, :], label=f'param_error_{i}')
        plt.xlabel('Experiment')
        plt.ylabel('param_error')
        min_indices = [np.argmin(row) for row in param_error]
        min_indices_str = ', '.join(map(str, min_indices))
        plt.title('min param_error: ' + min_indices_str + f' overall: {np.argmin(overall_param_error)}')
        plt.legend()
        plt.savefig(f'{sum_dir}param_error.png')
        with open(f'{sum_dir}param_error.pkl', 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if final_objects[0]['show_plots'] else None

    comm.Barrier()

    os.remove(f"{base_out_dir}results_{rank}.pkl")

    comm.Barrier()

    if tf_intra_op != 0:
        num_proc_grad = int(tf_intra_op)
    else:
        if p_m < p:
            print(f"cpu_count < p on rank {rank}.")
            return
        num_proc_grad = int(p_m / p)

    t = time()
    with Pool(processes=p) as pool:
        pool.map(partial(forward_add_optimization, num_proc_grad=num_proc_grad), final_obj_list[rank])

    comm.Barrier()

    if rank == 0 and len(final_obj_list) > 0 and final_obj_list[0][0]['optimization'] is True:
        print('Multi optimization time: {:.3f} s'.format(time() - t))

    comm.Barrier()


def forward_train_network(PrNN: dict) -> List[np.ndarray]:
    try:
        return train_network(PrNN)
    except Exception as e:
        # print('Error in train_network(): ' + str(e) + '\n' + traceback.format_exc())
        print('Error in train_network(): ' + str(e))
        return [np.array([]), ] * 3


def forward_add_optimization(PrNN: dict, num_proc_grad: int) -> None:
    """
    Adds optimization to a neural network based on specified parameters.

    Parameters:
    - PrNN (dict): Dictionary containing neural network parameters.
    - num_proc_grad (int): Number of processes for gradient computation.

    Returns:
    None
    """
    try:
        if PrNN['optimization']:
            PrOpt = ParametersOpt
            PrOpt['h_fixed'] = PrNN['h_fixed']
            # forward_func = ForwardHS_First
            forward_func = ForwardHS_Eff
            add_optimization(PrOpt, forward_func=forward_func, search_dir=PrNN['out_dir'], output_dir='',
                             num_proc=1, num_proc_grad=num_proc_grad)  # num_proc=1: only one output dir
    except Exception as e:
        # print('Error in add_optimization(): ' + str(e) + '\n' + traceback.format_exc())
        print('Error in add_optimization(): ' + str(e))
        return


def train_network(PrNN: dict) -> List[np.ndarray]:
    """
    Train a neural network model and perform predictions based on the given optimization parameters.

    Parameters:
    - PrNN (dict): Dictionary containing neural network training and prediction parameters.

    Returns:
    - List[np.ndarray]: A list containing computed errors and loss values.
    """
    if PrNN['training'] is False and PrNN['save_model'] is False:
        print('Error: training and save_model cannot be both False.')
        return [np.array([]), ] * 3
    PrNN['in_dir'] = PrNN['in_dir'] + '/' if PrNN['in_dir'] and not PrNN['in_dir'].endswith('/') else PrNN['in_dir']
    PrNN['out_dir'] = PrNN['out_dir'] + '/' if PrNN['out_dir'] and not PrNN['out_dir'].endswith('/') else PrNN[
        'out_dir']
    if PrNN['out_dir'] and not os.path.exists(PrNN['out_dir']):
        os.makedirs(PrNN['out_dir'])
    with open(f"{PrNN['out_dir']}ParametersNN.txt", 'w') as file:
        for key, value in PrNN.items():
            file.write(f'{key}: {value}\n')
    from src.NeuralNetwork import train, history, prediction
    if PrNN['training'] is True:
        model, scaler_input, scaler_output, model_history, train_valid_test_loss = train(PrNN)
        history(PrNN, model_history)
    else:
        model, scaler_input, scaler_output, model_history = None, None, None, None
        train_valid_test_loss = np.array([0., 0., 0.])  # get from saved model instead of zero

    # benchmarks
    par_vals = [np.array([200., 300., 400., 500., 6., 4., 8.]),
                np.array([300., 200., 400., 500., 6., 4., 8.]),
                np.array([300., 400., 200., 500., 6., 4., 8.])]

    cp_vals, w = [], np.empty([])
    for i in range(len(par_vals)):
        w, cp = ForwardHS_Eff(par_vals[i])
        cp_vals.append(cp)
    cp_vals = np.array(cp_vals)

    par_vals_per = prediction(PrNN, cp_vals, model, scaler_input, scaler_output)

    if PrNN['h_fixed'] == 0.:
        n = par_vals_per.shape[1] - 3  # Number of extra columns to add
        if n > 0:
            last_entries = par_vals_per[:, -1].copy()
            par_vals_per = np.pad(par_vals_per, ((0, 0), (0, n)), mode='constant', constant_values=0)
            par_vals_per[:, -n:] = last_entries[:, np.newaxis]
    elif PrNN['h_fixed'] == -1.:
        ...  # no change needed
    else:
        par_vals_per = np.pad(par_vals_per, ((0, 0), (0, par_vals_per.shape[1] - 1)), mode='constant',
                              constant_values=PrNN['h_fixed'])
    param_error = np.zeros(len(par_vals))
    for i in range(len(par_vals)):
        samples, labels = [par_vals[i], par_vals_per[i]], ['Ground truth', 'Prediction']
        plt.clf()
        velocities, thicknesses = None, None
        if len(par_vals[i]) == len(par_vals_per[i]):  # heights may not be the same
            param_error = np.mean(np.square(np.abs(par_vals[i] - par_vals_per[i]) / par_vals[i]))
        for idx, row in enumerate(samples):
            nr = (len(row) - 1) // 2
            velocities = np.array(row[:nr + 1])
            velocities = np.insert(velocities, 0, velocities[0])
            thicknesses = np.append(row[nr + 1:], 10)
            cumulative_thickness = np.cumsum(thicknesses)
            cumulative_thickness = np.insert(cumulative_thickness, 0, 0.)
            plt.step(velocities, -1 * cumulative_thickness, where='post', label=labels[idx])
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Depth (m)')
        plt.title(f'Velocity Profile_{str(velocities.astype(int))}_{str(thicknesses.astype(int))}')
        plt.legend()
        plt.grid(True)
        # plt.xlim(100, 600)
        # plt.ylim(0, np.max(samples[,:]))
        plt.savefig(f"{PrNN['out_dir']}prediction_profile_{i}.png")
        with open(f"{PrNN['out_dir']}prediction_profile_{i}.pkl", 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if PrNN['show_plots'] else None

    cp_per_vals = []
    cp_error = np.zeros(len(par_vals))
    for i in range(len(par_vals)):
        w, cp_per = ForwardHS_Eff(par_vals_per[i, :])
        cp_per_vals.append(cp_per)
        plt.clf()
        cp_error[i] = np.mean(np.square(np.abs(cp_per - cp_vals[i]) / cp_vals[i]))
        plt.plot(w, cp_vals[i], marker='o', label='Ground truth')
        plt.plot(w, cp_per, marker='o', label='Prediction')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase Velocity (m/s)')
        plt.legend()
        plt.savefig(f"{PrNN['out_dir']}prediction_cp_{i}.png")
        with open(f"{PrNN['out_dir']}prediction_cp_{i}.pkl", 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if PrNN['show_plots'] else None

    cp_per_vals = np.array(cp_per_vals)
    with open(f"{PrNN['out_dir']}prediction_benchmarks.pkl", "wb") as f:
        pickle.dump((w, par_vals, cp_vals, cp_per_vals, par_vals_per), f)

    return [cp_error, param_error, train_valid_test_loss]


def add_noise_to_cp_values(directories: List[str], noise_percentage: float) -> None:
    """
    Add Gaussian noise to the values in cp_values files within the specified directories.

    Parameters:
    - directories (List[str]): List of directory paths containing cp_values files.
    - noise_percentage (float): Percentage of Gaussian noise to add.

    Returns:
    - None
    """
    for directory in directories:
        # Check if the directory exists
        if not os.path.isdir(directory):
            print(f"Directory '{directory}' does not exist.")
            continue

        cp_values_file = os.path.join(directory, 'cp_values')
        if not os.path.isfile(cp_values_file):
            print(f"cp_values file not found in '{directory}'.")
            continue

        # Load the 2D array from cp_values file
        cp_values = np.loadtxt(cp_values_file)

        for i in range(cp_values.shape[0]):
            # noise = (noise_percentage / 100.) * np.random.normal(0, cp_values[i, :].std(), cp_values.shape[
            #     1])  # without randomizing the noise for each element
            noise = np.random.normal(0, (noise_percentage / 100.) * cp_values[i, :].std(), cp_values.shape[1])
            cp_values[i, :] = cp_values[i, :] + noise

        # Save the modified array with noise
        cp_values_noise_file = os.path.join(directory, f'cp_values_noise_{str(noise_percentage)}')
        np.savetxt(cp_values_noise_file, cp_values)

        print(f"Noise added to cp_values file in '{directory}' and saved as cp_values_noise.")


def generate_training(num_layer: int, num_sample: int, cs_range: List[float], h_range: List[float],
                      h_fixed: float = -1., max_scale_factor: float = 3., show_plots: bool = True,
                      out_dir: str = '', num_proc: int = 1) -> None:
    """
    Generate training samples and corresponding effective phase velocity values for a neural network.

    Parameters:
    - num_layer (int): Number of layers.
    - num_sample (int): Number of samples to generate.
    - cs_range (List[float]): Range of shear wave velocities for layers.
    - h_range (List[float]): Range of thicknesses for layers.
    - h_fixed (float, optional): Fixed thickness value if applicable (default is -1.).
               -1. means variable thickness for each layer
              = 1. means fixed thickness 1. for all layers
              = 0. means variable thickness but same for all layers
    - max_scale_factor (float, optional): Maximum scale factor for velocity/thickness differences (default is 3.).
    - show_plots (bool, optional): Whether to display plots (default is True).
    - out_dir (str, optional): Output directory to save generated samples and cp_values files (default is '').
    - num_proc (int, optional): Number of processes for parallel processing (default is 1).

    Returns:
    - None
    """
    # input ------------------------------
    m = num_sample
    n = int(num_layer)  # num_layer
    cpEffMax = Parameters.cpEffMax  # for filtering invalid cases
    cs_min = cs_range[0]
    cs_max = cs_range[1]
    h_min = h_range[0]
    h_max = h_range[1]
    w = Parameters.w
    generate = True
    plot_samples = False
    plot_cp = False
    if generate:
        samples = generate_samples(m, n, cs_min, cs_max, h_min, h_max, h_fixed, max_scale_factor)
        cp_values = parallel_process_samples(samples, w, num_proc)
        if h_fixed == 0.:
            samples = samples[:, :num_layer + 2]
        elif h_fixed == -1.:
            ...  # no change needed
        else:
            samples = samples[:, :num_layer + 1]
        # Filter out rows where all elements of cp_values are 0 or cpEffMax or any nan/inf in the row
        zero_rows = np.all(cp_values == 0., axis=1)
        cpEffMax_rows = np.all(cp_values == cpEffMax, axis=1)
        nan_inf_rows = np.any(np.isnan(cp_values) | np.isinf(cp_values), axis=1)
        invalid_rows = zero_rows | cpEffMax_rows | nan_inf_rows
        if np.any(invalid_rows):
            cp_values = cp_values[~invalid_rows]
            samples = samples[~invalid_rows]
        np.savetxt(f'{out_dir}samples', samples)
        np.savetxt(f'{out_dir}cp_values', cp_values)
        print(f'Layer {num_layer} | cp_size: {str(cp_values.shape)} - param_size: {str(samples.shape)}')
    else:
        samples = np.loadtxt(f'{out_dir}samples')
        cp_values = np.loadtxt(f'{out_dir}cp_values')
    if plot_samples or show_plots:
        if h_fixed == 0.:
            n = samples.shape[1] - 3  # Number of extra columns to add
            if n > 0:
                last_entries = samples[:, -1].copy()
                samples = np.pad(samples, ((0, 0), (0, n)), mode='constant', constant_values=0)
                samples[:, -n:] = last_entries[:, np.newaxis]
        elif h_fixed == -1.:
            ...  # no change needed
        else:
            samples = np.pad(samples, ((0, 0), (0, samples.shape[1] - 1)), mode='constant', constant_values=h_fixed)
        # samples = samples[1:2, :]
        for row in samples:
            velocities = np.array(row[:num_layer + 1])
            velocities = np.insert(velocities, 0, velocities[0])
            thicknesses = np.append(row[num_layer + 1:], 10)
            cumulative_thickness = np.cumsum(thicknesses)
            cumulative_thickness = np.insert(cumulative_thickness, 0, 0.)
            plt.step(velocities, -1 * cumulative_thickness, where='post')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Depth (m)')
        plt.title('Velocity Profile')
        plt.grid(True)
        plt.xlim(cs_min * 0.9, cs_max * 1.1)
        # plt.ylim(0, np.max(samples[,:]))
        plt.show()
    if plot_cp or show_plots:
        for cp_row in cp_values:
            plt.plot(w, cp_row)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Effective Phase Velocity (m/s)')
        plt.xlim(0, max(w))
        # plt.ylim(0, 1500)
        plt.box(True)
        plt.grid(True)
        plt.show()
        plt.clf()


def generate_training_multi_layer(n: List[int], num_sample: List[int], cs_range: List[float],
                                  h_range: List[float], h_fixed: float = -1.,
                                  max_scale_factor: float = 3., show_plots: bool = False,
                                  out_dir: str = '', num_proc: int = 1) -> None:
    """
    Generate training samples and corresponding effective phase velocity values for multiple layers using MPI.

    Parameters:
    - n (List[int]): Number of layers.
    - num_sample (List[int]): Total number of samples to generate for each layer.
    - cs_range (Tuple[float, float]): Range of shear wave velocities for layers.
    - h_range (Tuple[float, float]): Range of thicknesses for layers.
    - h_fixed (float, optional): Fixed thickness value if applicable (default is -1.).
               -1. means variable thickness for each layer
              = 1. means fixed thickness 1. for all layers
              = 0. means variable thickness but same for all layers
    - max_scale_factor (float, optional): Maximum scale factor for velocity/thickness differences (default is 3.).
    - show_plots (bool, optional): Whether to display plots (default is False).
    - out_dir (str, optional): Output directory to save generated samples and cp_values files (default is '').
    - num_proc (int, optional): Number of processes for parallel processing (default is 1).

    Returns:
    - None
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(num_sample) != len(n):
        print("n/num_sample in generate_training_multi_layer() has wrong size.") if rank == 0 else None
        return
    out_dir = out_dir + '/' if out_dir and not out_dir.endswith('/') else out_dir
    if rank == 0:
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for n_i in n:
            layer_out_dir = f'{out_dir}{n_i}/'
            if layer_out_dir and not os.path.exists(layer_out_dir):
                os.makedirs(layer_out_dir)

    comm.barrier()

    for ind, n_i in enumerate(n):
        if num_sample[ind] < size:
            continue
        parts = np.linspace(0, num_sample[ind], size + 1, dtype=int)
        parts = np.diff(parts)
        layer_out_dir = f'{out_dir}{n_i}/'
        rank_out_dir = f'{layer_out_dir}{rank}/'
        if rank_out_dir and not os.path.exists(rank_out_dir):
            os.makedirs(rank_out_dir)
        t = time()
        generate_training(num_layer=n_i, num_sample=int(parts[rank]), cs_range=cs_range, h_range=h_range,
                          h_fixed=h_fixed, max_scale_factor=max_scale_factor, show_plots=show_plots,
                          out_dir=rank_out_dir, num_proc=num_proc)

        comm.Barrier()

        if rank == 0:
            aggregated_cp_values = []
            aggregated_samples = []
            for i in range(0, size):
                rank_out_dir_i = f'{layer_out_dir}{i}/'
                cp_values = np.loadtxt(f'{rank_out_dir_i}cp_values')
                samples = np.loadtxt(f'{rank_out_dir_i}samples')
                aggregated_cp_values.append(cp_values)
                aggregated_samples.append(samples)
                print(f'>>> Layer {n_i} | Portion {i}, cp_size: {cp_values.shape}, param_size: {samples.shape}')
            aggregated_cp_values = np.vstack(aggregated_cp_values)
            aggregated_samples = np.vstack(aggregated_samples)
            print(f'>>> Layer {n_i} | Final, cp_size: {aggregated_cp_values.shape}, '
                  f'param_size: {aggregated_samples.shape}')
            indices = np.random.permutation(aggregated_samples.shape[0])
            aggregated_cp_values = aggregated_cp_values[indices]
            aggregated_samples = aggregated_samples[indices]

            np.savetxt(f'{layer_out_dir}cp_values', aggregated_cp_values)
            np.savetxt(f'{layer_out_dir}samples', aggregated_samples)
            print(f'>>> Layer {n_i} | Time:', time() - t)

        comm.Barrier()

        if os.path.exists(rank_out_dir) and os.path.isdir(rank_out_dir):
            shutil.rmtree(rank_out_dir)

        comm.Barrier()

    layer_out_dir = f'{out_dir}{n[0]}/'
    if rank == 0 and len(n) == 1 and os.path.exists(f'{layer_out_dir}cp_values') and os.path.exists(
            f'{layer_out_dir}samples'):
        os.remove(f'{out_dir}cp_values') if os.path.exists(f'{out_dir}cp_values') else None
        os.remove(f'{out_dir}samples') if os.path.exists(f'{out_dir}samples') else None
        sleep(1)
        shutil.move(f'{layer_out_dir}cp_values', out_dir)
        shutil.move(f'{layer_out_dir}samples', out_dir)
        sleep(3)
        shutil.rmtree(layer_out_dir)

    comm.Barrier()


def load_and_show_plots(directory: str) -> None:
    """Load and display plots saved in .pkl files within the specified directory.

    Args:
        directory (str): The path to the directory containing .pkl files.

    Returns:
        None
    """
    files = os.listdir(directory)

    pkl_files = [file for file in files if file.endswith('.pkl')]

    if not pkl_files:
        print("No .pkl files found in the directory.")
        return

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        with open(file_path, 'rb') as f:
            loaded_plot = pickle.load(f)

        # Show the loaded plot
        plt.figure(loaded_plot.number)
        plt.show()


def add_optimization_process_directory(directory: str, output_dir: str, PrOpt: dict, forward_func: callable,
                                       num_proc_grad: int) -> None:
    """
    Adds optimization to a process directory based on specified parameters.

    Parameters:
    - directory (str): Directory containing data for optimization.
    - output_dir (str): Output directory for saving optimization results.
    - PrOpt (dict): Dictionary containing optimization parameters.
    - forward_func (callable): Callable function for forward computation.
    - num_proc_grad (int): Number of processes for gradient computation.

    Returns:
    None
    """
    file_name = 'prediction_benchmarks.pkl'
    with open(os.path.join(directory, file_name), "rb") as f:
        w, par_vals, cp_vals, cp_per_vals, par_vals_per = pickle.load(f)
    cp_optimized_all, optimized_param_all = [], []
    for i in range(len(par_vals)):
        PrOpt['optim_id'] = i
        if output_dir == '':
            PrOpt['out_dir'], output_dir_i = directory, directory
        else:
            PrOpt['out_dir'] = output_dir_i = f"{output_dir}{os.path.basename(os.path.normpath(directory))}/"
        par_vals_per_i = par_vals_per[i, :]
        optimized_param = WaveDispOptim(PrOpt=PrOpt, forward_func=forward_func, cp_observed=cp_vals[i, :],
                                        init_params=par_vals_per_i, n_proc=num_proc_grad)
        _, cp_optimized = forward_func(optimized_param)
        cp_optimized_all.append(cp_optimized)
        optimized_param_all.append(optimized_param)
        plt.clf()
        plt.plot(w, cp_vals[i, :], label='Grand truth')
        plt.plot(w, cp_optimized, label='Optimized', linestyle='dotted', linewidth=3)
        plt.plot(w, cp_per_vals[i, :], label='Predicted')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase Velocity (m/s)')
        plt.box(True)
        plt.legend()
        plt.savefig(f"{output_dir_i}optim_cp_{i}.png")
        with open(f"{output_dir_i}optim_cp_{i}.pkl", 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if PrOpt['show_plots'] else None

        n = len(optimized_param)
        if PrOpt['h_fixed'] == 0.:
            optimized_param = np.pad(optimized_param, (0, n - 3), mode='constant', constant_values=optimized_param[-1])
            par_vals_per_i = np.pad(par_vals_per_i, (0, n - 3), mode='constant', constant_values=par_vals_per_i[-1])
        elif PrOpt['h_fixed'] == -1.:
            ...  # no change needed
        else:
            optimized_param = np.pad(optimized_param, (0, n - 1), mode='constant', constant_values=PrOpt['h_fixed'])
            par_vals_per_i = np.pad(par_vals_per_i, (0, n - 1), mode='constant', constant_values=PrOpt['h_fixed'])

        plt.clf()
        samples = [par_vals[i], optimized_param, par_vals_per_i]
        labels = ['Ground truth', 'Optimized', 'Predicted']
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
        plt.savefig(f"{output_dir_i}optim_profile_{i}.png")
        with open(f"{output_dir_i}optim_profile_{i}.pkl", 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show() if PrOpt['show_plots'] else None
    cp_optimized_all, optimized_param_all = np.array(cp_optimized_all), np.array(optimized_param_all)
    with open(f"{directory}optim_prediction_benchmarks.pkl", 'wb') as f:
        pickle.dump([w, par_vals, cp_vals, cp_per_vals, par_vals_per, cp_optimized_all, optimized_param_all], f)


def add_optimization(PrOpt: dict, forward_func: callable, search_dir: str, output_dir: str = '',
                     num_proc: int = 1, num_proc_grad: int = 1) -> None:
    """
    Adds optimization based on specified parameters.

    Parameters:
    - PrOpt (dict): Dictionary containing optimization parameters.
    - forward_func (callable): Callable function for forward computation.
    - search_dir (str): Directory to search for optimization data.
    - output_dir (str, optional): Output directory for saving optimization results (default is '').
    - num_proc (int, optional): Number of processes for optimization (default is 1).
    - num_proc_grad (int, optional): Number of processes for gradient computation (default is 1).

    Returns:
    None

    Notes:
    - If output_dir is not specified or an empty string, it defaults to the same value as search_dir.
    """
    output_dir = output_dir + '/' if output_dir and not output_dir.endswith('/') else output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    directories = []
    file_name = 'prediction_benchmarks.pkl'
    for root, dirs, files in os.walk(search_dir):
        if file_name in files:
            root = root.replace('\\', '/')
            root += '/' if not root.endswith('/') else None
            directories.append(root)
    if num_proc == 1:
        for directory in directories:
            add_optimization_process_directory(directory=directory, output_dir=output_dir, PrOpt=PrOpt,
                                               forward_func=forward_func, num_proc_grad=num_proc_grad)
    else:
        p = cpu_count(logical=False)  # num Physical processes
        p = len(directories) if 0 < len(directories) < p else p
        p = num_proc if 0 < num_proc < p else p
        process_directory = partial(add_optimization_process_directory, output_dir=output_dir, PrOpt=PrOpt,
                                    forward_func=forward_func, num_proc_grad=num_proc_grad)
        with Pool(processes=p) as pool:
            pool.map(process_directory, directories)

import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import copy
from itertools import product
import scipy.optimize
import math
from functools import reduce
import random

def generate_param_combinations(param_dict):
    keys = param_dict.keys()
    for combination in product(*param_dict.values()):
        yield dict(zip(keys, combination))

def generate_filtered_param_combinations(param_dict, filter_func):
    keys = param_dict.keys()
    for combination in product(*param_dict.values()):
        candidate = dict(zip(keys, combination))
        if filter_func(candidate):
            yield candidate

def get_resource_units_bellalta_policy(channel_bandwidth, num_users):
    if channel_bandwidth == 20e6 and num_users >= 9:
        return 9
    elif channel_bandwidth == 40e6 and num_users >= 18:
        return 18
    elif channel_bandwidth == 80e6 and num_users >= 37:
        return 37
    return num_users

def get_resource_units_roundrobin_policy(channel_bandwidth, num_users):
    if channel_bandwidth == 20e6:
        if num_users >= 9:
            return 9
        elif num_users >= 4:
            return 4
        elif num_users >= 2:
            return 2
        elif num_users >= 1:
            return 1
    elif channel_bandwidth == 40e6:
        if num_users >= 18:
            return 18
        elif num_users >= 8:
            return 8
        elif num_users >= 4:
            return 4
        elif num_users >= 2:
            return 2
        elif num_users >= 1:
            return 1
    elif channel_bandwidth == 80e6:
        if num_users >= 37:
            return 37
        elif num_users >= 16:
            return 16
        elif num_users >= 8:
            return 8
        elif num_users >= 4:
            return 4
        elif num_users >= 2:
            return 2
        elif num_users >= 1:
            return 1
    elif channel_bandwidth == 160e6:
        if num_users >= 74:
            return 74
        elif num_users >= 32:
            return 32
        elif num_users >= 16:
            return 16
        elif num_users >= 8:
            return 8
        elif num_users >= 4:
            return 4
        elif num_users >= 2:
            return 2
        elif num_users >= 1:
            return 1
    if num_users == 0:
        return 1
    raise ValueError("Unrecognized channel width and number of users combination")

def get_data_subcarrier_count(channel_bandwidth, num_resource_units):
    if channel_bandwidth == 20e6:
        if num_resource_units <= 1:
            return 234
        elif num_resource_units <= 2:
            return 102
        elif num_resource_units <= 4:
            return 48
        elif num_resource_units <= 9:
            return 24
    elif channel_bandwidth == 40e6:
        if num_resource_units <= 1:
            return 468
        elif num_resource_units <= 2:
            return 234
        elif num_resource_units <= 4:
            return 102
        elif num_resource_units <= 8:
            return 48
        elif num_resource_units <= 18:
            return 24
    elif channel_bandwidth == 80e6:
        if num_resource_units <= 1:
            return 980
        elif num_resource_units <= 2:
            return 468
        elif num_resource_units <= 4:
            return 234
        elif num_resource_units <= 8:
            return 102
        elif num_resource_units <= 16:
            return 48
        elif num_resource_units <= 37:
            return 24
    elif channel_bandwidth == 160e6:
        if num_resource_units <= 1:
            return 2 * 980
        elif num_resource_units <= 2:
            return 980
        elif num_resource_units <= 4:
            return 468
        elif num_resource_units <= 8:
            return 234
        elif num_resource_units <= 16:
            return 102
        elif num_resource_units <= 32:
            return 48
        elif num_resource_units <= 74:
            return 24

def get_modulation_details(mcs_index):
    if not isinstance(mcs_index, int) or not (0 <= mcs_index <= 11):
        raise ValueError("MCS index must be an integer between 0 and 11 inclusive.")
    bits_per_symbol = [1, 2, 2, 4, 4, 6, 6, 6, 8, 8, 10, 10]
    code_rate = [1/2, 1/2, 3/4, 1/2, 3/4, 2/3, 3/4, 5/6, 3/4, 5/6, 3/4, 5/6]
    return [bits_per_symbol[mcs_index], code_rate[mcs_index]]

def calculate_model_throughput(params, verbose=False):
    mcs_index = 11 if params['mcs'] == 'ideal' else params['mcs']
    bits_per_symbol, coding_rate = get_modulation_details(mcs_index)
    num_stations = params['nStations']
    frame_bits = params['frameSize'] * 8
    aggregated_frames = params['Na']
    min_cw_ap = 15e1000 if params['dl'] == 'None' else params['cwMin']
    min_cw_station = 15e1000 if params['ul'] == 'None' else params['cwMin']

    backoff_stages_ap = backoff_stages_sta = 6
    symbol_duration = 12.8e-6 + 800e-9
    empty_slot_time = 9e-6
    channel_bandwidth = params['channelWidth'] * 1e6

    ru_number_function = get_resource_units_roundrobin_policy if params['scheduler'] == 'rr' else get_resource_units_bellalta_policy
    num_resource_units = max(1, min(num_stations, ru_number_function(channel_bandwidth, num_stations)))
    data_subcarriers_per_ru = get_data_subcarrier_count(channel_bandwidth, ru_number_function(channel_bandwidth, num_stations))
    data_subcarriers_su = get_data_subcarrier_count(channel_bandwidth, ru_number_function(channel_bandwidth, 1))

    spatial_streams = 1
    bits_per_ofdm_symbol_mu = spatial_streams * bits_per_symbol * coding_rate * data_subcarriers_per_ru
    bits_per_ofdm_symbol_su = spatial_streams * bits_per_symbol * coding_rate * data_subcarriers_su
    max_txop_duration_sec = params['maxTxopDuration'] * 1e-6

    def calculate_mu_dl_ampdu_length(num_aggregated_frames_mu_dl):
        if params['ackSeqType'] == 'ACK-SU-FORMAT':
            return num_aggregated_frames_mu_dl * (32 + frame_bits + 208)
        elif params['ackSeqType'] == 'MU-BAR':
            return num_aggregated_frames_mu_dl * (32 + frame_bits + 208)
        elif params['ackSeqType'] == 'AGGR-MU-BAR':
            return num_aggregated_frames_mu_dl * (32 + frame_bits + 208) + (32 + 9 * num_resource_units + 16)

    def calculate_mu_dl_duration(num_aggregated_frames_mu_dl):
        ampdu_length = calculate_mu_dl_ampdu_length(num_aggregated_frames_mu_dl)
        return 44e-6 + 16e-6 + np.ceil(ampdu_length / bits_per_ofdm_symbol_mu) * symbol_duration

    def calculate_su_dl_duration(num_aggregated_frames_su):
        ampdu_length_su = num_aggregated_frames_su * (32 + frame_bits + 208)
        return 44e-6 + np.ceil(ampdu_length_su / bits_per_ofdm_symbol_su) * symbol_duration

    num_aggregated_frames_su = 1
    while True:
        if num_aggregated_frames_su >= aggregated_frames or calculate_su_dl_duration(num_aggregated_frames_su) > max_txop_duration_sec:
            num_aggregated_frames_su -= 1
            break
        num_aggregated_frames_su += 1

    su_dl_throughput = ((32 * num_aggregated_frames_su * frame_bits) / (calculate_su_dl_duration(num_aggregated_frames_su) + empty_slot_time)) / 1e6
    mu_dl_throughput = su_dl_throughput / num_stations
    hol_delay = num_stations * (calculate_su_dl_duration(num_aggregated_frames_su) + 15 * empty_slot_time) * 1000

    if verbose:
        print(f"SU DL Throughput: {su_dl_throughput} Mbps")
        print(f"MU DL Throughput: {mu_dl_throughput} Mbps")
        print(f"Head-of-Line Delay: {hol_delay} ms")

    return [mu_dl_throughput, su_dl_throughput, hol_delay]

def run_model_throughput_calculation(validation_params):
    params_with_metrics = copy.deepcopy(validation_params)
    params_with_metrics['metrics'] = ['mu_dl', 'su_dl', 'hol']
    model_results = xr.DataArray(
        np.zeros([len(param) for param in params_with_metrics.values()]),
        list(zip(params_with_metrics.keys(), params_with_metrics.values()))
    )
    with Pool() as pool:
        param_combinations = list(generate_param_combinations(validation_params))
        for param_combination, output in zip(
            param_combinations, pool.imap(calculate_model_throughput, tqdm(param_combinations, total=len(param_combinations), desc="Running model", unit="param set"))
        ):
            model_results.loc[param_combination] = output
    return model_results

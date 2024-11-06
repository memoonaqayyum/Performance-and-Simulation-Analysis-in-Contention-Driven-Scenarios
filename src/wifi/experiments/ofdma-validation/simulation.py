import numpy as np
import copy
import sem
import matplotlib.pyplot as plt

def adjust_simulation_parameters(param_set, step=4, sim_duration=1, flow_rate=100):
    adjusted_params = copy.deepcopy(param_set)
    adjusted_params['dlFlowDataRate'] = [flow_rate]
    adjusted_params['ulFlowDataRate'] = [flow_rate]
    adjusted_params['simulationTime'] = [sim_duration]
    adjusted_params['nEhtStations'] = [0]
    adjusted_params['nStations'] = list(range(param_set['nStations'][0], param_set['nStations'][-1] + 1, step))
    return adjusted_params

def modify_simulation_parameters(param_set, step=4, sim_duration=1, flow_rate=100, additional_parameters=None):
    if not isinstance(param_set, dict):
        raise ValueError("param_set must be a dictionary.")
    
    modified_params = copy.deepcopy(param_set)
    modified_params['dlFlowDataRate'] = [flow_rate]
    modified_params['ulFlowDataRate'] = [flow_rate]
    modified_params['simulationTime'] = [sim_duration]
    
    if 'nStations' in param_set:
        modified_params['nEhtStations'] = [0]
        modified_params['nStations'] = list(range(param_set['nStations'][0], param_set['nStations'][-1] + 1, step))
    else:
        raise KeyError("'nStations' key not found in param_set.")
    
    if additional_parameters and isinstance(additional_parameters, dict):
        modified_params.update(additional_parameters)
    
    return modified_params

def visualize_with_confidence_intervals(data_results, num_runs, x_coord, plot_multiple=False):
    squeezed_results = data_results.squeeze()
    std_dev = np.transpose(squeezed_results.reduce(np.std, 'runs').data)
    avg_values = np.transpose(squeezed_results.reduce(np.mean, 'runs').data)
    confidence_interval = 2.576 * std_dev / np.sqrt(num_runs)
    
    ax = plt.gca()
    median_properties = dict(linestyle='None')
    box_properties = dict(linestyle='None')
    whisker_properties = dict(linewidth=1)
    
    if len(avg_values.shape) > 1:
        for i in range(len(avg_values)):
            for index, coord_value in enumerate(squeezed_results.coords[x_coord].data):
                box_stats = {
                    "med": avg_values[i][index],
                    "q1": avg_values[i][index],
                    "q3": avg_values[i][index],
                    "whislo": avg_values[i][index] - confidence_interval[i][index],
                    "whishi": avg_values[i][index] + confidence_interval[i][index]
                }
                ax.bxp([box_stats], positions=[coord_value], showfliers=False,
                       medianprops=median_properties, boxprops=box_properties, whiskerprops=whisker_properties)
    else:
        for index, coord_value in enumerate(squeezed_results.coords[x_coord].data):
            box_stats = {
                "med": avg_values[index],
                "q1": avg_values[index],
                "q3": avg_values[index],
                "whislo": avg_values[index] - confidence_interval[index],
                "whishi": avg_values[index] + confidence_interval[index]
            }
            ax.bxp([box_stats], positions=[coord_value], showfliers=False,
                   medianprops=median_properties, boxprops=box_properties, whiskerprops=whisker_properties)

def create_simulation_campaign(param_set, num_runs, allow_overwrite):
    campaign_manager = sem.CampaignManager.new('../../../..', 'wifi-ofdma-validation',
                                                'ofdma-validation-results',
                                                runner_type='ParallelRunner',
                                                overwrite=allow_overwrite,
                                                optimized=True, check_repo=False,
                                                max_parallel_processes=1)
    simulation_params = copy.deepcopy(param_set)
    simulation_params['verbose'] = [False]
    campaign_manager.run_missing_simulations(simulation_params, num_runs)
    return campaign_manager

def extract_simulation_metrics(sim_result):
    metrics_lines = iter(sim_result['output']['stdout'].splitlines())
    dl_throughput = ul_throughput = dl_legacy = ul_legacy = hol_delay = dl_mu_complete = he_tb_complete = 0
    dl_rates = ul_rates = []

    try:
        while True:
            line = next(metrics_lines, None)
            if line is None:
                break

            if "Per-AC" in line:
                next(metrics_lines)
                dl_rates.append(float(next(metrics_lines).split()[-1]))
                next(metrics_lines)
                ul_value = next(metrics_lines).split()[-1]
                ul_rates.append(float(ul_value) if ul_value else 0)

            if "Throughput (Mbps) [DL]" in line:
                dl_throughput = extract_total_value(metrics_lines)
            elif "Throughput (Mbps) [UL]" in line:
                ul_throughput = extract_total_value(metrics_lines)
            elif "Throughput (Mbps) [DL] LEGACY" in line:
                dl_legacy = extract_total_value(metrics_lines)
            elif "Throughput (Mbps) [UL] LEGACY" in line:
                ul_legacy = extract_total_value(metrics_lines)
            elif "Pairwise Head-of-Line delay" in line:
                hol_delay = extract_hol_value(metrics_lines)
            elif "DL MU PPDU completeness" in line or "HE TB PPDU completeness" in line:
                dl_mu_complete = he_tb_complete = 0  # Placeholder or logic if needed

        return [dl_throughput, ul_throughput, dl_legacy, ul_legacy, hol_delay, dl_mu_complete, he_tb_complete]
    
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return [0, 0, 0, 0, 0, 0, 0]

def extract_total_value(lines_iterator):
    while True:
        line = next(lines_iterator, None)
        if line is None or "TOTAL:" in line:
            return float(line.split()[-1]) if line else 0

def extract_hol_value(lines_iterator):
    while True:
        line = next(lines_iterator, None)
        if line is None:
            return 0
        if "TOTAL:" in line:
            count_value = float(line.split("[")[1].split("]")[0])
            return float(line.split("<")[1].split(">")[0]) if count_value != 0 else 0

def collect_simulation_results(param_set, num_runs=2, allow_overwrite=False, output_folder='validation-results',
                               detailed_output=False, extra_metrics=None, custom_campaign_manager=None):
    
    base_metrics = ['dl', 'ul', 'dllegacy', 'ullegacy', 'hol', 'dlmucomp', 'hetbcomp']
    all_metrics = base_metrics + (extra_metrics if extra_metrics and isinstance(extra_metrics, list) else [])

    campaign_manager = custom_campaign_manager if custom_campaign_manager else sem.CampaignManager.new(
        '../../../../', 'wifi-ofdma-validation', output_folder,
        runner_type='ParallelRunner', overwrite=allow_overwrite,
        check_repo=False, max_parallel_processes=1
    )

    simulation_params = copy.deepcopy(param_set)
    simulation_params['verbose'] = [detailed_output]

    if num_runs is None:
        campaign_manager.run_missing_simulations(simulation_params)
    else:
        campaign_manager.run_missing_simulations(simulation_params, num_runs)

    if detailed_output:
        for result in campaign_manager.db.get_results(simulation_params):
            print(sem.utils.get_command_from_result('wifi-ofdma-validation', result))

    sim_results_array = campaign_manager.get_results_as_xarray(
        simulation_params, extract_simulation_metrics, all_metrics, num_runs).squeeze()

    return sim_results_array

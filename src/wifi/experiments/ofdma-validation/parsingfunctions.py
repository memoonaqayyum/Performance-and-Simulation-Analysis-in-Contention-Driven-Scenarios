import numpy as np
from io import StringIO
import itertools
import xarray as xr
import pandas as pd
import seaborn as sns
import copy
import sem
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import ast

sns.set_style("whitegrid")

def extract_flow_statistics(simulation_result):
    flow_entries = [item for item in simulation_result['output'].items() if "flowStats" in item[0]]
    flow_list = []
    flow_index = 1

    for filename, content in flow_entries:
        flow_data = content.splitlines()[0].split(" ")
        latencies = content.splitlines()[1].split(" ")[:-1] if len(content.splitlines()) > 1 else []
        parsed_latencies = [float(latency) for latency in latencies if len(latencies) > 1]

        flow_info = {
            "flowIndex": flow_index,
            "bssIndex": int(flow_data[0]),
            "staId": int(flow_data[1]),
            "direction": flow_data[2],
            "accessCategory": flow_data[3],
            "transportType": flow_data[4],
            "throughput": float(flow_data[5]),
            "expectedRate": float(flow_data[6]),
            "actualRate": float(flow_data[7]),
            "packetLossRate": float(flow_data[8]),
            "droppedPackets": int(flow_data[9]),
            "txBytes": int(flow_data[10]),
            "txPackets": int(flow_data[11]),
            "rxBytes": int(flow_data[12]),
            "rxPackets": int(flow_data[13]),
            "latencySamples": parsed_latencies
        }
        flow_list.append(flow_info)
        flow_index += 1

    return flow_list

def retrieve_flow_information(simulation_result):
    parsed_flows = extract_flow_statistics(simulation_result)
    if not parsed_flows:
        print(sem.utils.get_command_from_result('wifi-multi-ap', simulation_result))
        return [float('nan')] * 30
    return [list(flow.values())[:-1] for flow in parsed_flows]

def collect_flow_latencies(simulation_result):
    parsed_flows = extract_flow_statistics(simulation_result)
    return [[flow['flowIndex'], flow['bssIndex'], flow['accessCategory'], flow['direction'], flow['transportType'], latency] 
            for flow in parsed_flows for latency in flow['latencySamples']]

def extract_pairwise_statistics(simulation_result):
    statistics_list = []

    for key in simulation_result['output']:
        if "pairwise" in key:
            data_content = simulation_result['output'][key]
            expired, rejected, failed = [float(value) for value in data_content.splitlines()[0].split(" ")]
            l2_latencies = [float(value) for value in data_content.splitlines()[1].split(" ")[:-1]]
            pairwise_hol = [float(value) for value in data_content.splitlines()[2].split(" ")[:-1]]
            ampdu_sizes = [float(value) for value in data_content.splitlines()[3].split(" ")[:-1]]
            ampdu_ratios = [float(value) for value in data_content.splitlines()[4].split(" ")[:-1]]

            parsed_entry = {
                'bssIndex': int(key.split("-")[1]),
                'direction': key.split("-")[2].upper(),
                'staId': int(key.split("-")[3]),
                'accessCategory': key.split("-")[4].upper(),
                'expiredPackets': [expired],
                'rejectedPackets': [rejected],
                'failedPackets': [failed],
                'l2Latencies': l2_latencies,
                'pairwiseHol': pairwise_hol,
                'ampduSizes': ampdu_sizes,
                'ampduRatios': ampdu_ratios
            }
            statistics_list.append(parsed_entry)

    return statistics_list

def extract_per_ac_statistics(simulation_result):
    statistics = []

    for key in simulation_result['output']:
        if "perac" in key:
            data_content = simulation_result['output'][key]
            queue_dropped = [float(value) for value in data_content.splitlines()[0].split(" ")]
            txop_durations = [float(value) for value in data_content.splitlines()[1].split(" ")[:-1]]
            queue_sojourn_times = [float(value) for value in data_content.splitlines()[2].split(" ")[:-1]]
            aggregate_hol = [float(value) for value in data_content.splitlines()[3].split(" ")[:-1]]

            parsed_entry = {
                'bssIndex': int(key.split("-")[1]),
                'staId': int(key.split("-")[2]),
                'accessCategory': key.split("-")[3].upper(),
                'queueDroppedPackets': [queue_dropped],
                'txopDurations': txop_durations,
                'queueSojournTimes': queue_sojourn_times,
                'aggregateHol': aggregate_hol
            }
            statistics.append(parsed_entry)

    return statistics

def collect_per_ac_samples(simulation_result, target_quantity):
    parsed_stats = extract_per_ac_statistics(simulation_result)
    return [[stat['bssIndex'], stat['staId'], stat['accessCategory'], value] 
            for stat in parsed_stats for value in stat[target_quantity]]

def collect_aggregate_statistics(simulation_result, target_stat):
    aggregated_stats = extract_per_bss_statistics(simulation_result, target_stat)
    return [[entry[0], value] for entry in aggregated_stats for value in entry[1]]

def extract_phy_states(simulation_result):
    phy_entries = []

    for line in simulation_result['output']['WifiPhyStateLog.txt'].splitlines():
        timestamp, context_id, start_time, duration, state = line.split(" ")
        if state == "TX":
            phy_entries.append([int(context_id), int(start_time), int(duration), state])

    return phy_entries

def visualize_phy_states(simulation_result, output_filename=None):
    state_colors = {
        "RX": 'r', "TX": 'b', "IDLE": 'grey', "CCA_BUSY": 'grey',
        "SWITCHING": 'grey', "SLEEP": 'grey', "OFF": 'grey'
    }
    active_states = {"TX": True, "RX": False, "IDLE": False, "CCA_BUSY": False, "SWITCHING": False, "SLEEP": False, "OFF": False}

    for line in simulation_result['output']['WifiPhyStateLog.txt'].splitlines():
        timestamp, context_id, start_time, duration, state = line.split(" ")
        if active_states[state]:
            plt.plot([float(start_time), float(start_time) + float(duration)], 
                     [float(context_id), float(context_id)], state_colors[state], linewidth=2)

    for line in simulation_result['output']['txops.txt'].splitlines():
        bss_id, device_id, context_id, start_time, duration = line.strip().split(" ")
        plt.plot([int(start_time), int(start_time) + int(duration)], [int(context_id), int(context_id)], 
                 'grey', linewidth=20, solid_capstyle='butt', alpha=0.3)

    num_devices_per_bss = simulation_result['params']['nEhtStations'] + simulation_result['params']['nHeStations'] + 1
    num_bss = simulation_result['params']['nBss']

    ap_count = 1
    sta_count = 1
    for idx in range(num_devices_per_bss * num_bss):
        device_type = f"AP{ap_count}" if idx % num_devices_per_bss == 0 else f"STA{sta_count}"
        if idx % num_devices_per_bss == 0:
            ap_count += 1
            sta_count = 1
        else:
            sta_count += 1
        plt.text(0.3 * float(duration), idx, device_type, fontsize=12, va='center')

    plt.title("Device State Visualization")
    plt.xlabel("Time [ns]")
    plt.ylim(-1, num_devices_per_bss * num_bss + 1)
    plt.grid(axis='x', which='both')
    plt.gca().spines['left'].set_visible(False)

    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

def visualize_device_locations(simulation_result, output_filename=None):
    color_palette = list(itertools.islice(iter(cm.rainbow(np.linspace(0, 1, simulation_result['params']['nBss']))), 
                                          simulation_result['params']['nBss']))
    plt.figure()
    for line in simulation_result['output']['locations.txt'].splitlines():
        bss_id, device_id, x_coord, y_coord = line.split(" ")
        plt.text(float(x_coord), float(y_coord), device_id, color=color_palette[int(bss_id)])
    
    plt.ylim(-20, 20)
    plt.xlim(-20, 20)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Network Topology Visualization")

    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_cw_bo_statistics(simulation_result):
    bo_log = pd.DataFrame([[int(value) for value in entry.split(" ")] 
                           for entry in simulation_result['output']['BackoffLog.txt'].splitlines()], 
                          columns=['timestamp', 'deviceId', 'backoffValue'])
    
    cw_log = pd.DataFrame([[int(value) for value in entry.split(" ")] 
                           for entry in simulation_result['output']['CWLog.txt'].splitlines()], 
                          columns=['timestamp', 'deviceId', 'previousCw', 'currentCw']).drop(columns='previousCw')
    
    cw_log['type'] = 'cw'
    bo_log['type'] = 'bo'
    combined_logs = pd.concat([cw_log, bo_log], ignore_index=True)

    sns.relplot(data=combined_logs, x='timestamp', y='currentCw', row='deviceId', style='type', kind='line')

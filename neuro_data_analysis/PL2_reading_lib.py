import os
import sys 
sys.path.append("D:\Poncelab_Github\OmniPlex and MAP Offline SDK Bundle\Python 3 PL2 Offline Files SDK")
dll_directory = r"D:\Poncelab_Github\OmniPlex and MAP Offline SDK Bundle\Python 3 PL2 Offline Files SDK\bin"
if sys.version_info >= (3, 8):
    os.add_dll_directory(dll_directory)
os.environ['PATH'] = dll_directory + os.pathsep + os.environ.get('PATH', '')
from pypl2 import pl2_ad, pl2_spikes, pl2_events, pl2_info, pl2_comments
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt

def read_pl2_info(filename):
    output = pl2_info(filename)
    if output == 0:
        print(f"Error reading {filename}")
        raise ValueError(f"Error reading {filename}")
    spkinfo, evtinfo, adinfo = output
    return spkinfo, evtinfo, adinfo


def read_channel_unit_info(filename, threshold=10):
    output = pl2_info(filename)
    if output == 0:
        print(f"Error reading {filename}")
        raise ValueError(f"Error reading {filename}")
    spkinfo, evtinfo, adinfo = output

    spike_channel_id = []
    spike_unit_id = []
    spike_total_cnt = []
    for i in range(len(spkinfo)):
        for j in range(len(spkinfo[i].units)):
            spike_cnt = spkinfo[i].units[j]
            if spike_cnt > threshold:
                spike_channel_id.append(spkinfo[i].channel)
                spike_unit_id.append(j)
                spike_total_cnt.append(spike_cnt)

    spike_channel_id = np.array(spike_channel_id)
    spike_unit_id = np.array(spike_unit_id)
    spike_total_cnt = np.array(spike_total_cnt)

    return spike_channel_id, spike_unit_id, spike_total_cnt


def plot_unit_waveforms(filename, channel_id, unit_id):
    spkinfo, evtinfo, adinfo = pl2_info(filename)
    figh, axs = plt.subplots(8, 8, figsize=(20, 20))
    axs = axs.flatten()
    for i in trange(len(spkinfo)):
        plt.sca(axs[i])
        channel_id = spkinfo[i].channel
        spikes = pl2_spikes(filename, channel_id - 1) # here it is 0-indexed, so we need to subtract 1
        uniq_units = np.unique(spikes.units)
        for iunit in range(len(uniq_units)):
            unit_id = uniq_units[iunit] # 0, 1, 2, 3,   0 is unsorted, 1 is unit A
            spk_unit_0 = spikes.units == unit_id
            ts_unit_0 = np.array(spikes.timestamps)[spk_unit_0]
            wave_unit_0 = np.array(spikes.waveforms)[spk_unit_0]
            avg_waveform = np.mean(wave_unit_0, axis=0)
            plt.plot(avg_waveform, label=f'Unit {unit_id} [N={len(ts_unit_0)}]')
        plt.title(f'Channel {channel_id} ')
        plt.legend()
    plt.tight_layout()
    plt.show()

import os
import pickle
from sys import float_info
import warnings

import torch
import numpy as np
from tqdm import tqdm
from torchinfo import summary

from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.signal import savgol_filter, argrelextrema, find_peaks, peak_prominences
from KDEpy import FFTKDE


class NeuronProfiler:

    def __init__(self, network, dataloader, device, layer_type=torch.nn.Conv2d, save_name=None, load_if_exists=False, keep_cumulative=False):

        self.device = device

        self.save_name = save_name
        self.load_if_exists = load_if_exists

        self.network = network
        self.dataloader = dataloader

        self.average_threshold = 0
        self.average_threshold_n = 0

        sample_image_size = list(self.dataloader)[0][0].shape

        summary_list = summary(network, verbose=False, input_size=sample_image_size).summary_list
        self.layer_list = [layer for layer in summary_list if (layer.is_leaf_layer and isinstance(layer.module,  layer_type))]
        self.number_of_layers = len(self.layer_list)

        self.handles = []

        self.neuron_max = []
        self.neuron_min = []
        for layer in self.layer_list:
            self.neuron_max.append((torch.ones(layer.output_size, dtype=torch.float) * float_info.min).to(self.device))
            self.neuron_min.append((torch.ones(layer.output_size, dtype=torch.float) * float_info.max).to(self.device))

        self.keep_cumulative = keep_cumulative
        if self.keep_cumulative:
            self.neuron_cumulative = [[] for _ in range(self.number_of_layers)]

    def return_max_min_hook_function(self, layer_index, keep_cumulative=False):
        def hook(module, input, output):
            with torch.no_grad():
                torch.maximum(self.neuron_max[layer_index], output, out=self.neuron_max[layer_index])
                torch.minimum(self.neuron_min[layer_index], output, out=self.neuron_min[layer_index])

                if keep_cumulative:
                    self.neuron_cumulative[layer_index].append(output.detach().cpu())
            return

        return hook

    def append_dataset_toggle_activity_hook(self, keep_cumulative=False):
        for layer_index, layer in enumerate(self.layer_list):
            self.handles.append(layer.module.register_forward_hook(self.return_max_min_hook_function(layer_index, keep_cumulative=keep_cumulative)))

    def remove_hooks(self):
        """
        Remove alle the hooks from the network
        """
        for handle in self.handles:
            handle.remove()

    def profile_network(self):

        try:

            if not(self.save_name is not None and self.load_if_exists):
                raise FileNotFoundError

            print(f'Trying to load profiling from saved files...', flush=True)
            self.neuron_max = torch.load(open(f'Profiler/{self.save_name}_max.pkl', 'rb'), map_location=self.device)
            self.neuron_min = torch.load(open(f'Profiler/{self.save_name}_min.pkl', 'rb'), map_location=self.device)

            if self.keep_cumulative:
                warnings.warn(f'WARNING: Loading neuron cumulative values from file is not supported. If you want to '
                              f'compute the neuron cumulative values set load_if_exist to False.')
                # self.neuron_cumulative = pickle.load(open(f'Profiler/{self.save_name}_min.pkl', 'rb'))

            print(f'Profiling loaded from file', flush=True)

        except FileNotFoundError:

            print(f'No file to load from, starting profiler...', flush=True)

            with torch.no_grad():

                pbar = tqdm(self.dataloader)

                self.remove_hooks()
                self.append_dataset_toggle_activity_hook(self.keep_cumulative)

                self.network.to(self.device)
                self.network.eval()

                for n, entry in enumerate(pbar):
                    golden_data = entry[0].to(self.device)
                    self.network(golden_data)

                if self.save_name is not None:
                    os.makedirs(f'Profiler', exist_ok=True)
                    torch.save(self.neuron_max, f'Profiler/{self.save_name}_max.pkl')
                    torch.save(self.neuron_min, f'Profiler/{self.save_name}_min.pkl')

                    # if keep_cumulative:
                    #     pickle.dump(self.neuron_cumulative, open(f'Profiler/{self.save_name}_cumulative.pkl', 'wb'))

    def get_binary_threshold(self, split_value=0.5, mode='global'):

        if mode == 'global':
            max_value = np.max([t.max().cpu() for t in self.neuron_max])
            min_value = np.min([t.min().cpu() for t in self.neuron_min])
            threshold = (max_value + min_value) * split_value
        elif mode == 'layer':
            max_value = np.array([float(t.max().cpu()) for t in self.neuron_max])
            min_value = np.array([float(t.min().cpu()) for t in self.neuron_min])
            threshold = (max_value + min_value) * split_value
        elif mode == 'neuron':
            threshold = [(self.neuron_max[i] + self.neuron_min[i]) * split_value for i in range(self.number_of_layers)]
        else:
            exit('Invalid mode for threshold, exiting program...')

        return threshold

    def get_ternary_threshold(self, split_value_low=0.10, split_value_high=0.90, mode='global'):

        if mode == 'global':
            max_value = np.max([t.max().cpu() for t in self.neuron_max])
            min_value = np.min([t.min().cpu() for t in self.neuron_min])
            threshold_low = (max_value + min_value) * split_value_low
            threshold_high = (max_value + min_value) * split_value_high
        elif mode == 'layer':
            max_value = np.array([float(t.max().cpu()) for t in self.neuron_max])
            min_value = np.array([float(t.min().cpu()) for t in self.neuron_min])
            threshold_low = (max_value + min_value) * split_value_low
            threshold_high = (max_value + min_value) * split_value_high
        elif mode == 'neuron':
            threshold_low = [(self.neuron_max[i] + self.neuron_min[i]) * split_value_low for i in range(self.number_of_layers)]
            threshold_high = [(self.neuron_max[i] + self.neuron_min[i]) * split_value_high for i in range(self.number_of_layers)]
        else:
            exit('Invalid mode for threshold, exiting program...')

        return threshold_low, threshold_high

    def get_kde_threshold(self):
        threshold_list = []
        for layer_index in range(self.number_of_layers):
            flattened_neuron_values = [self.neuron_cumulative[layer_index][i].flatten() for i in
                                       range(len(self.neuron_cumulative[layer_index]))]
            threshold = []
            output_shape = self.neuron_cumulative[layer_index][0].size()
            for neuron_index in tqdm(range(len(flattened_neuron_values[0]))):
                neuron_values = [float(flattened_neuron_values[i][neuron_index]) for i in
                                 range(len(self.neuron_cumulative[layer_index]))]
                # plt.hist(neuron_values, density=True, bins=50, alpha=.75)
                # plt.axvline(max(neuron_values)*.25, linestyles='-.', c='orange', label='25% threshold')
                # plt.axvline(max(neuron_values)*.75, linestyles='--', c='orange', label='75% threshold')
                # plt.ylim(0)
                # plt.ylabel('Probability Density [%]')
                # plt.xlim(0)
                # plt.xlabel('Value')
                # plt.title(f'Output probability of sample neuron #{neuron_index}')

                if np.count_nonzero(neuron_values) == 0:
                    threshold.append(0)
                else:
                    try:
                        x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(np.array(neuron_values)).evaluate(100)
                        y = savgol_filter(y, window_length=7, polyorder=3)
                        # plt.plot(x, y, c='b', label='KDE')
                        mi = argrelextrema(y, np.less)[0]
                        peaks = find_peaks(y)[0]

                        if len(y[mi]) > 0:
                            peaks_prominence = peak_prominences(y, peaks)
                            # Threshold is the lowest point between the two most prominent peaks
                            most_prominent_peaks_indices = np.sort(peaks[np.argpartition(peaks_prominence[0], -2)[-2:]])
                            minimum_between_peaks_indices = mi[np.where((x[mi] > x[most_prominent_peaks_indices[0]]) & (
                                        x[mi] < x[most_prominent_peaks_indices[1]]))]
                            threshold_index = minimum_between_peaks_indices[np.argmin(y[minimum_between_peaks_indices])]
                            threshold_x = x[threshold_index]
                            threshold_y = y[threshold_index]
                            threshold.append(threshold_x)
                            # plt.scatter(x[most_prominent_peaks_indices], y[most_prominent_peaks_indices], color='green', label='Peak', zorder=10)
                            # plt.scatter(threshold_x, threshold_y, color='r', zorder=10, label='Threshold')
                            # plt.scatter(threshold_x, threshold_y, color='r', zorder=10, label='Threshold')
                        else:
                            threshold.append(0)
                        # plt.legend()
                        # plt.show()
                        pass
                    except ValueError:
                        threshold.append(0)

            print(len(threshold))
            threshold_torch = torch.tensor(threshold)
            threshold_torch = threshold_torch.reshape(output_shape)
            threshold_list.append(threshold_torch)

        return threshold_list

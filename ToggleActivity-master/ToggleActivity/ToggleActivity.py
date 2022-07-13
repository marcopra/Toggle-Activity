import copy

import pandas
import torch
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import pandas as pd


class ToggleActivity:

    def __init__(self,
                 golden_network,
                 golden_dataloader,
                 binary_threshold=None,
                 ternary_threshold_low=None,
                 ternary_threshold_high = None,
                 toggle_network=None,
                 toggle_dataloader=None,
                 toggle_dataset_function=None,
                 layer_type = torch.nn.Conv2d,
                 device='cpu'):
        """
        The class used to measure the toggle activity of a network. It can work either to compute the dataset toggle
        activity (the total number of neurons that have all the possible quantized values over a dataset) or to compute
        the average toggle activity of a
        :param golden_network: The network for which to measure the toggle activity.
        :param golden_dataloader: The dataloader for which to measure the toggle activity.
        :param toggle_network: Default None. Used only to measure the toggle activity of tuples of images. If specified,
        use this network to perform the prediction over the image of the toggle dataset. If None, create a deepcopy of the
        golden network to be used as toggle_network.
        :param toggle_dataloader: Default None. Used only to measure the toggle activity of tuples of images. This is the
        dataset containing the image that have a high toggle activity compared to the one in the golden data_loader. It
        is assumed that, given an index i, a tuple of images that have a high toggle activity are <golden_dataloader[i],
        toggle_dataloader[i]>. If None, uses the golden_data loader.
        :param toggle_dataset_function: Default function. Used only to measure the toggle activity of tuples of images.
        This function is applied to the images fed to the toggle network.
        """
        self.device = device

        self.golden_network = golden_network.to(self.device)
        self.toggle_network = copy.deepcopy(self.golden_network.to(self.device)) if toggle_network is None else toggle_network

        sample_image_size = list(golden_dataloader)[0][0].shape

        golden_summary_list = summary(golden_network, verbose=False, input_size=sample_image_size).summary_list
        golden_summary_list_leaf = [layer for layer in golden_summary_list if (layer.is_leaf_layer and isinstance(layer.module, layer_type))]
        self.golden_layer_list = [layer.module for layer in golden_summary_list if (layer.is_leaf_layer and isinstance(layer.module, layer_type))]
        self.number_of_layers = len(self.golden_layer_list)

        toggle_summary_list = summary(self.toggle_network, verbose=False).summary_list
        self.toggle_layer_list = [layer.module for layer in toggle_summary_list if layer.is_leaf_layer]

        self.golden_neuron_values = [None] * self.number_of_layers
        self.toggle_neuron_values = [None] * self.number_of_layers

        self.golden_handles = []
        self.toggle_handles = []

        self.golden_dataloader = golden_dataloader
        self.toggle_dataloader = self.golden_dataloader if toggle_dataloader is None else toggle_dataloader

        self.toggle_dataloader = toggle_dataloader
        self.toggle_dataset_function = toggle_dataset_function

        if binary_threshold is None and (ternary_threshold_low is None or ternary_threshold_high is None):
            raise ValueError('Either a binary threshold or both ternary threshold should be set...')

        self.binary_threshold = binary_threshold
        self.ternary_threshold_low = ternary_threshold_low
        self.ternary_threshold_high = ternary_threshold_high

        self.dataset_toggle_activity_list = []
        for i in range(self.number_of_layers):
            self.dataset_toggle_activity_list.append([torch.zeros(int(np.prod(golden_summary_list_leaf[i].output_size))).to(self.device),
                                                      torch.zeros(int(np.prod(golden_summary_list_leaf[i].output_size))).to(self.device)])

    def __save_ternary_values_hook(self, layer_index):
        def hook(model, input, output):

            flattened_output = output.squeeze().to(self.device).flatten()

            if isinstance(self.ternary_threshold_low, float) and isinstance(self.ternary_threshold_high, float):
                threshold_low = self.ternary_threshold_low
                threshold_high = self.ternary_threshold_high
            else:
                if isinstance(self.ternary_threshold_low[layer_index], float) and isinstance(self.ternary_threshold_high[layer_index], float):
                    threshold_low = self.ternary_threshold_low[layer_index]
                    threshold_high = self.ternary_threshold_high[layer_index]
                else:
                    threshold_low = self.ternary_threshold_low[layer_index].flatten().to(self.device)
                    threshold_high = self.ternary_threshold_high[layer_index].flatten().to(self.device)

            over_threshold = flattened_output > threshold_high
            under_threshold = flattened_output <= threshold_low

            self.dataset_toggle_activity_list[layer_index][0] += under_threshold.int()
            self.dataset_toggle_activity_list[layer_index][1] += over_threshold.int()

        return hook

    def __save_binary_values_hook(self, layer_index):
        def hook(model, input, output):

            flattened_output = output.squeeze().to(self.device).flatten()

            if isinstance(self.binary_threshold, float):
                threshold = self.binary_threshold
            else:
                if isinstance(self.binary_threshold[layer_index], float):
                    threshold = self.binary_threshold[layer_index]
                else:
                    threshold = self.binary_threshold[layer_index].flatten().to(self.device)

            over_threshold = flattened_output > threshold
            under_threshold = flattened_output <= threshold

            self.dataset_toggle_activity_list[layer_index][0] += under_threshold.int()
            self.dataset_toggle_activity_list[layer_index][1] += over_threshold.int()

        return hook

    def save_value_hook(self, layer_index, golden=True):
        """
        Returns a hook that saves the value of the  neurons of the specified layer
        :param layer_index: The index of the layer to which the hook will be attached to
        :param golden: Default True. Whether to save the golden output or the toggle output
        :return: A hook to attach to layer layer_index
        """
        def hook(model, input, output):
            new_value = output.squeeze().to(self.device)
            if golden:
                self.golden_neuron_values[int(layer_index)] = new_value
            else:
                self.toggle_neuron_values[int(layer_index)] = new_value

        return hook

    def __append_dataset_toggle_activity_hook(self, mode='binary'):

        if mode != 'binary' and mode != 'ternary':
            raise ValueError('Invalid mode, choose one of binary or ternary')
        
        for layer_index, layer in enumerate(self.golden_layer_list):
            if mode == 'binary':
                self.golden_handles.append(layer.register_forward_hook(self.__save_binary_values_hook(layer_index)))
            else:
                self.golden_handles.append(layer.register_forward_hook(self.__save_ternary_values_hook(layer_index)))


    def append_toggle_activity_hooks(self, golden=True):
        """
        Append hooks for saving the value to all the layers of the network 
        :param golden: Default True. Whether to append hooks for the golden layer or the toggle layer.
        """
        if golden:
            for layer_index, layer in enumerate(self.golden_layer_list):
                self.golden_handles.append(layer.register_forward_hook(self.save_value_hook(layer_index, golden)))
        else:
            for layer_index, layer in enumerate(self.toggle_layer_list):
                self.toggle_handles.append(layer.register_forward_hook(self.save_value_hook(layer_index, golden)))

    def remove_hooks(self, golden=True):
        """
        Remove alle the hooks from the network
        :param golden: Default True. Whether to remove hooks from the golden or the toggle network.
        """
        handles = self.golden_handles if golden else self.toggle_handles
        for handle in handles:
            handle.remove()

    def compute_dataset_toggle_activity(self, mode='binary'):
        """
        Compute the dataset toggle activity: the percentage of neurons that have all the possible quantized values over a dataset
        :param mode: Either 'binary' or 'threshold'. Default 'binary'. For 'binary', check whether the value of each
        neuron is at least once over and at least once under the binary threshold, for 'ternary' check whether the vale
        of each neuron is at least once under the lower ternary threshold and at least once the higher ternary threshold.
        :return: A tuple dataset toggle activity, list of dataset toggle activity per layer
        """

        if mode != 'binary' and mode != 'ternary':
            raise ValueError('Invalid mode, choose one of binary or ternary')

        with torch.no_grad():

            pbar = tqdm(self.golden_dataloader)
            pbar.set_description(f'{mode.capitalize()} dataset toggle activity')

            self.remove_hooks()
            self.__append_dataset_toggle_activity_hook(mode=mode)

            for n, entry in enumerate(pbar):
                golden_data = entry[0].to(self.device)

                golden_prediction = torch.argmax(self.golden_network(golden_data))

            toggled_neurons_per_layer = np.array([])
            neurons_per_layer = np.array([])

            for layer_toggle_activity in self.dataset_toggle_activity_list:
                neuron_toggle_activity = np.array([bool(neuron[0] > 0 and neuron[1] > 0) for neuron in zip(*layer_toggle_activity)])
                toggled_neurons_per_layer = np.append(toggled_neurons_per_layer, neuron_toggle_activity.sum())
                neurons_per_layer = np.append(neurons_per_layer, len(neuron_toggle_activity))

            dataset_toggle_activity_per_layer = toggled_neurons_per_layer / neurons_per_layer
            dataset_toggle_activity = toggled_neurons_per_layer.sum() / neurons_per_layer.sum()

            return dataset_toggle_activity, dataset_toggle_activity_per_layer

    def compute_toggle_activity(self, save_detailed_results=True, save_summarized_results=False):
        """
        Compute the toggle activity of the golden network when compared with the toggle network. If a toggle dataset has
        been specified during the initialization use that for the toggle network, otherwise use the golden dataset.
        :param save_detailed_results: Default True. Whether to return a pandas dataframe containing the toggle activity
        for each layer of all the images in the test set
        :param save_summarized_results: Default False. Whether to return a pandas dataframe containing the toggle
        activity for each image in the test set
        :return: If save_detailed_results or save_summarized_results, returns a panda dataset with the toggle activity.
        """

        if save_summarized_results and save_detailed_results:
            raise AttributeError('Only one of save_summarized_results and save_detailed_results can be declared True')

        golden_accuracy_running_mean = 0
        toggle_activity_running_mean = 0
        pbar = tqdm(self.golden_dataloader)

        toggle_list = []

        for n, entry in enumerate(pbar):
            golden_data = entry[0]
            golden_label = entry[1]

            if self.toggle_dataloader is not None:
                toggle_entry = next(self.toggle_dataloader)
                toggle_data = toggle_entry[0]
                toggle_label = toggle_entry[1]
            else:
                toggle_data = golden_data
                toggle_label = golden_label

            self.remove_hooks(golden=True)
            self.append_toggle_activity_hooks(golden=True)
            golden_prediction = torch.argmax(self.golden_network(golden_data))

            self.remove_hooks(golden=False)
            self.append_toggle_activity_hooks(golden=False)
            toggle_data = self.toggle_dataset_function(toggle_data) if self.toggle_dataset_function is not None else toggle_data
            toggle_prediction = torch.argmax(self.toggle_network(toggle_data.float()))

            toggled_count = 0
            untoggled_count = 0
            for layer_index in range(self.number_of_layers):
                if self.golden_neuron_values[layer_index] is not None:
                    toggled = torch.sign(self.golden_neuron_values[layer_index]) == -1 * torch.sign(self.toggle_neuron_values[layer_index])

                    layer_toggled_count = int(toggled.int().sum())
                    toggled_count += layer_toggled_count

                    layer_untoggled_count = len(toggled.flatten(0)) - layer_toggled_count
                    untoggled_count += layer_untoggled_count

                    layer_toggle_activity = toggled_count / (toggled_count + untoggled_count)

                    if save_detailed_results:
                        toggle_list.append([n, layer_index, f'{layer_toggle_activity:.2f}'])

            toggle_activity = toggled_count/(toggled_count + untoggled_count)

            if save_summarized_results:
                toggle_list.append([n, f'{toggle_activity:.2f}'])

            golden_accuracy_running_mean = (golden_accuracy_running_mean * n + float(golden_prediction == golden_label)) / (n + 1)
            toggle_activity_running_mean = (toggle_activity_running_mean * n + toggle_activity)/(n + 1)
            pbar.set_postfix({'Toggle Activity': f'{toggle_activity_running_mean * 100:.2f}%',
                              'Golden Accuracy': f'{golden_accuracy_running_mean * 100:.2f}%'})

        if save_detailed_results or save_summarized_results:
            columns = ['ImageIndex', 'ToggleActivity'] if save_summarized_results else ['ImageIndex', 'LayerIndex', 'ToggleActivity']
            toggle_dataframe = pd.DataFrame(toggle_list, columns=columns)
            return toggle_dataframe

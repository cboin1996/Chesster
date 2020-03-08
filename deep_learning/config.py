import os, sys
import json
"""
Parameters used in the construction of the models conv2D layers
"""
class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = False
    input_depth = 18
    p_filter_size = 2
    p_kernel_size = 1
    v_filter_size = 4
    v_kernel_size = 1


class Config:
    def __init__(self):
        self.model = ModelConfig()
        with open(os.path.join(sys.path[0],  'uci.json'), 'r') as f:
            uci_data = json.load(f)
        self.labels = uci_data['moves']
        self.num_labels = len(self.labels)

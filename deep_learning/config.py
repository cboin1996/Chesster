import os, sys
import json
import numpy as np
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

class PlayerConfig:
    """
    Config for the player module and self_play

    """
    def __init__(self):
        self.c_puct = 1.5 # potential next move trade off
        self.dirichlet_alpha = 0.3 # noise parameter
        self.tau_decay_rate = 0.99 # temperature
        self.virtual_loss = 3
        self.search_threads = 16
        self.simulation_num_per_move = 100 # numbers of sims in MCTS
        self.resign_threshold = -0.8 # score to resign play at.
        self.min_resign_turn = 5 # number of turns before resigning is allowed
        self.noise_eps = 0.25
        self.max_processes = 3 # number of model threads
        self.max_game_length = 1000
        self.thinking_loop = 1
        self.logging_thinking = False

class TrainerConfig:
    """
    Config for the trainer

    """
    def __init__(self):
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 5 # RAM explosion...
        self.vram_frac = 1.0
        self.batch_size = 384 # tune this to your gpu memory
        self.dataset_size = 300000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.loss_weights = [1.25, 1.0] # [policy, value] prevent value overfit in SL
        self.epoch_to_checkpoint = 3



class PlayWithHumanConfig:
    """
    Config for allowing human to play against an agent using uci

    """
    def __init__(self):
        self.simulation_num_per_move = 1200
        self.threads_multiplier = 2
        self.c_puct = 1 # lower  = prefer mean action value
        self.noise_eps = 0
        self.tau_decay_rate = 0  # start deterministic mode
        self.resign_threshold = None

    def update_play_config(self, pc):
        """
        :param PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.search_threads *= self.threads_multiplier
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.tau_decay_rate = self.tau_decay_rate
        pc.resign_threshold = self.resign_threshold
        pc.max_game_length = 999999

        return pc

class PlayDataConfig:
    """
    Data parameters

    """
    def __init__(self):
        self.min_elo_policy = 500 # 0 weight
        self.max_elo_policy = 1800 # 1 weight
        self.sl_nb_game_in_file = 250
        self.nb_game_in_file = 50
        self.max_file_num = 150

class EvaluateConfig:
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 13 # number of games to play before updating model
        self.replace_rate = 0.55
        self.player_conf = PlayerConfig()
        self.player_conf.simulation_num_per_move = 100
        self.player_conf.thinking_loop = 1
        self.player_conf.c_puct = 1 # lower  = prefer mean action value
        self.player_conf.tau_decay_rate = 0.6 # I need a better distribution...
        self.player_conf.noise_eps = 0
        self.evaluate_latest_first = True
        self.max_game_length = 1000

class Options:
    new = False


class Resources:
    def __init__(self):
        dir = os.path.dirname
        self.project_dir = dir(dir(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_dir, 'deep_learning', 'best_model')
        self.model_dir = os.path.join(self.project_dir, 'deep_learning', 'best_model')
        self.model_best_conf_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.uci_data_path = os.path.join(self.project_dir, 'deep_learning', 'uci.json')

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "deep_learning", "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

        self.lichess_settings_path = os.path.join(self.project_dir, "secret.json")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)

def flip_uci_labels(uci_labels):
    def replace(label):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in label])

    return [replace(label) for label in uci_labels]

class Config:
    with open(os.path.join(Resources().uci_data_path), 'r') as f:
        uci_data = json.load(f)
    labels = uci_data['moves']
    num_labels = int(len(labels))
    flipped_labels = flip_uci_labels(labels)
    unflipped_index = None
    def __init__(self):
        self.opts = Options()
        self.model = ModelConfig()
        self.eval = EvaluateConfig()
        self.player_conf = PlayerConfig()
        self.resource = Resources()
        self.play_data = PlayDataConfig()
        self.trainer = TrainerConfig()


        self.labels = Config.labels
        self.flipped_labels = Config.flipped_labels
        self.num_labels = Config.num_labels


    """
    Arguments: pol policy to flip:
    return: the policy, flipped (for switching between black and white)
    """
    @staticmethod
    def flip_policy(pol):

        return np.asarray([pol[ind] for ind in Config.unflipped_index])

Config.unflipped_index = [Config.labels.index(x) for x in Config.flipped_labels]

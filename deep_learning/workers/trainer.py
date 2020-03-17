import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle

import numpy as np

from deep_learning.agent.model import ChessModel
from deep_learning.agent.player import Player
from deep_learning.agent.model_api import ChessAPI
from deep_learning.lib.data_ops import get_game_data_filepaths, read_game_data_from_file, get_next_generation_model_dirs
from deep_learning.lib.model_ops import load_best_model_weight
from deep_learning.environment import canon_input_planes, is_black_turn, testeval
from deep_learning.config import Config

from keras.optimizers import Adam
from keras.callbacks import TensorBoard

logger = getLogger(__name__)

def start(config: Config):
    return Trainer(config).start()

class Trainer:
    """
    Worker which optomizes the chess model by training it on data.
    attributes:
        config: the configuration to us
        model: the chess model
        dataset: tuple of dequeues where each dequeue contains game states,
            target policy network values (calculated based on visit stats
                for each state during the game), and target value network values (calculated based on
                    who actually won the game after that state)
        executor: the thread executor for running training processes.
    """


    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = deque(), deque(), deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)


    def start(self):

        self.model = self.load_model()
        self.training()

    def training(self):
        """
        Does the actual training of the model running it on game data endlessy...
        """
        self.compile_model()
        self.filenames = deque(get_game_data_filepaths(self.config.resource))
        shuffle(self.filenames)
        total_steps = self.config.trainer.start_total_steps

        while True:
            self.fill_queue()
            steps = self.training_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            self.save_current_model()
            state, policy, value = self.dataset
            while len(state) > self.config.trainer.dataset_size/2:
                state.popleft()
                policy.popleft()
                value.popleft()

    def training_epoch(self, epochs):
        """
        Runs some number of epochs of training
        arguments:
            epochs: number of epochs
        return:
            number of datapoint that were trained in total baby
        """
        trainer_config = self.config.trainer
        state_array, policy_array, value_array = self.collect_all_loaded_data()
        tensorboard_callback = TensorBoard(log_dir="./logs", batch_size=trainer_config.batch_size, histogram_freq=1)
        self.model.model.fit(state_array, [policy_array, value_array],
                             batch_size=trainer_config.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02,
                             callbacks=[tensorboard_callback])
        steps = (state_array.shape[0] // trainer_config.batch_size) * epochs
        return steps

    def compile_model(self):
        """
        Compiles the model to use optimizer and loss func tuned for supervised learning baby
        """
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised
        self.model.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def save_current_model(self):
        """
        Compiles the model to use a optimizer and loss function tuned for supervised learning
        """
        resource = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(resource.next_generation_model_dir, resource.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, resource.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, resource.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def collect_all_loaded_data(self):
        """
        return: a tuple containing the data in self.dataset, split into
        (state, policy, and value).
        """
        state_array,policy_array,value_array=self.dataset

        state_array1 = np.asarray(state_array, dtype=np.float32)
        policy_array1 = np.asarray(policy_array, dtype=np.float32)
        value_array1 = np.asarray(value_array, dtype=np.float32)
        return state_array1, policy_array1, value_array1

    def fill_queue(self):
        """
        Fills the self.dataset attribute with data from the training data set loaded.
        """
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                logger.debug(f"loading the data from {filename}")
                futures.append(executor.submit(load_data_from_file, filename))

            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    logger.debug(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file, filename))

    def load_model(self):
        """
        Loads the next generation model from the directory or the best model if next isnt found.
        """
        model = ChessModel(self.config)
        resource = self.config.resource
        dirs = get_next_generation_model_dirs(resource)

        if not dirs:
            logger.debug("loading the best model")
            if not load_best_model_weight(model):
                raise RuntimeError("Cannot load the best Model!")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading latest model")
            config_path = os.path.join(latest_dir, resource.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, resource.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    return convert_to_cheating_data(data)

def convert_to_cheating_data(data):
    """
    argument: data: format is SelfPlayWorker.buffer
    """
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:

        state_planes = canon_input_planes(state_fen)

        if is_black_turn(state_fen):
            policy = Config.flip_policy(policy)

        move_number = int(state_fen.split(' ')[5])
        value_certainty = min(5, move_number)/5 # reduces the noise of the opening... plz train faster
        sl_value = value*value_certainty + testeval(state_fen, False)*(1-value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list, dtype=np.float32)

"""
 Worker that saves play data to files
"""

from deep_learning.agent.model import ChessModel
from deep_learning.config import Config
from deep_learning.agent.player import Player
from deep_learning.environment import Chess, Victor
from deep_learning.lib.data_ops import pretty_print, write_game_data_to_file
from deep_learning.lib.model_ops import reload_best_model_weight_if_changed, load_best_model_weight, save_as_best_model

from multiprocessing import Manager
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from time import time
from threading import Thread

import os

from logging import getLogger

logger = getLogger(__name__)

def start(config: Config):
    return WorkerForSelfPlay(config).start()

class WorkerForSelfPlay:
    def __init__(self, config:Config):
        self.config = config
        self.curr_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.curr_model.get_pipes(self.config.player_conf.search_threads) for _ in range(self.config.player_conf.max_processes)])
        self.buffer = []

    def start(self):
        """
            Perform self play while writing data to files
        """
        self.buffer = []

        futures = deque()

        with ProcessPoolExecutor(max_workers=self.config.player_conf.max_processes) as executor:
            for game_index in range(self.config.player_conf.max_processes * 2):
                futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes))
            game_index = 0

            while True:
                game_index += 1
                start_time = time()
                env, data = futures.popleft().result()
                print(f"game {game_index:3} time={time() - start_time:5.1f}s "
                      f"halfturns={env.num_halfturns:3} {env.victor:12} "
                      f"{'by resign' if env.isResigned else '          '}")

                pretty_print(env, ("current_model", "current_model"))
                self.buffer += data
                if (game_index % self.config.play_data.nb_game_in_file) == 0:
                    self.flush_buffer()
                    reload_best_model_weight_if_changed(self.curr_model)
                futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes))

        if len(data) > 0:
            self.flush_buffer()
    def load_model(self):
        """
        Loads the chess model into self.curr_model
        """
        model = ChessModel(self.config)

        if model.config.opts.new or not load_best_model_weight(model):
            model.build_model()
            save_as_best_model(model)
        return model

    def flush_buffer(self):
        logger.debug("Beginning a save")
        resource = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(resource.play_data_dir, resource.play_data_filename_tmpl % game_id)
        logger.debug(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

def self_play_buffer(config, cur) -> (Chess, list):
    """
    Play a game and add its data to the buffer
    Arguments:
        config: play cofig
        list(connection) cur: a list of pipes to send the observsations to for getting predictions from the model
    return: (Chess, list((str,list(float)): a tuple containig the final Chess state and a list of data for the buffer
    """
    pipes = cur.pop()
    env = Chess().reinit()

    white = Player(config, pipes=pipes)
    black = Player(config, pipes=pipes)

    while not env.over():
        # env.print_pretty()
        if env.whites_turn():
            action = white.action(env)
        if env.blacks_turn():
            action = black.action(env)
        env.make_move(action)
        if env.num_halfturns >= config.player_conf.max_game_length:
            env.choose_victor()

    if env.victor ==  Victor.white:
        black_win = -1
    elif env.victor == Victor.black:
        black_win = 1
    else:
        black_win = 0

    black.finish_game(black_win)
    white.finish_game(-black_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    cur.append(pipes)
    return env, data

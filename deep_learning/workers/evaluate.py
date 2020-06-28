"""
The worker which will play sets of games against next generation models and pick the best one
"""

import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep

from deep_learning.agent.model import ChessModel
from deep_learning.agent.player import Player
from deep_learning.config import Config
from deep_learning.environment import Chess, Victor
from deep_learning.lib.data_ops import get_next_generation_model_dirs, pretty_print
from deep_learning.lib.model_ops import save_as_best_model, load_best_model_weight

logger = getLogger(__name__)

def start(config: Config):
    return Evaluator(config).start()

class Evaluator:
    """
    Worker which evaluates traubed models keeping track of the best one
    Attributes:
        Config config: program config to use
        PlayerConfig play_conf: tje playconfig taken from config.eval.player_conf
        ChessModel self.curr_model: the chess model to use
        Manager manager: multiprocessing manager
        list(Connection) cur_pipes: the pipes on which the current best ChessModel is listening to make predictions about games
    """
    def __init__(self, config:Config):
        """
        Arguments:
            config: the config to control how evaluation works
        """
        self.config = config
        self.play_conf = config.eval.player_conf
        self.curr_model = self.load_current_model()
        self.manager = Manager()
        self.cur_pipes = self.manager.list([self.curr_model.get_pipes(self.play_conf.search_threads) for _ in range(self.play_conf.max_processes)])

    def start(self):
        """
        Starts the evaluation, loading best models from directory that stores them
         and chgecks if they perform better than the current best
        """
        while True:
            nextgen_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluating model {model_dir}")
            is_nextgen_better = self.evaluate_model(nextgen_model)
            if is_nextgen_better:
                logger.debug(f"New model has become the best: {model_dir}")
                save_as_best_model(nextgen_model)
                self.curr_model = nextgen_model
            self.move_model(model_dir)
    def evaluate_model(self, nextgen_model):
        """
        Evaluates nextgen_model by playing it against the current best for games given by config.eval.game_num

        Arguments:
            ChessModel nextgen_model: the model the evaluate against the best
        Returns true if nextgen_model is better than current best
        """
        nextgen_pipes = self.manager.list([nextgen_model.get_pipes(self.play_conf.search_threads) for _ in range(self.play_conf.max_processes)])
        futures = []
        with ProcessPoolExecutor(max_workers=self.play_conf.max_processes) as executor:
            for game_index in range(self.config.eval.game_num):
                fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=nextgen_pipes, current_white=(game_index % 2 == 0))
                futures.append(fut)

            results = []
            for fut in as_completed(futures):
                nextgen_score, env, current_white = fut.result()
                results.append(nextgen_score)

                win_rate = sum(results)/len(results)
                game_index = len(results)

                logger.debug(f"game {game_index:3}: nextgen_score={nextgen_score:.1f} as {'black' if current_white else 'white'}"
                             f"{'by resign ' if env.isResigned else '            '}"
                             f"win_rate={win_rate*100:5.1f}% "
                             f"{env.board.fen().split(' ')[0]}")
                colors = ("curr_model", "nextgen_model")
                if not current_white:
                    colors = reversed(colors)
                pretty_print(env, colors)

                if len(results)-sum(results) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                    logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                    return False
                if sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    logger.debug(f"win count has reached {results.count(1)} so change best model")
                    return True
        win_rate = sum(results) / len(results)
        logger.debug(f"winning rate {win_rate*100:.1f}%")
        return win_rate >= self.config.eval.replace_rate

    def move_model(self, model_dir):
        """
        Moves the newest model to the specified directory

        Arguments:
            str model_dir: directory where model should be moved
        """
        model_dirname = model_dir.split('/')[-1]
        resource = self.config.resource
        new_dir = os.path.join(resource.next_generation_model_dir, "copies", model_dirname)
        os.rename(model_dir, new_dir)

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        return: ChessModel: the model
        """
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        """
        Loads the next generation model from the standard directory
        return: (ChessModel, file): the model and the directory that it was in
        """
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info("There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = ChessModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

def play_game(config, cur, ng, current_white: bool) -> (float, Chess, bool):
    """
    Plays a game against models cur and ng and reports the results
    Arguments:
        Config config: config for how to play
        list(Connection) cur: the current model
        list(Connection) nextgen: should be nextgen model
        bool current_white: determines whether cur should be white or black
    return: (float, Chess, bool): the score (0 loss, 0.5 draw 1 win), environment and bool which is true if cur was white in the game
    """
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()
    env = Chess().reinit()

    current_player = Player(config, pipes=cur_pipes, play_conf=config.eval.player_conf)
    nextgen_player = Player(config, pipes=ng_pipes, play_conf=config.eval.player_conf)

    if current_white:
        white, black = current_player, nextgen_player
    else:
        white, black = nextgen_player, current_player

    while not env.over():
        if env.whites_turn():
            action = white.action(env)
        else:
            action = black.action(env)
        env.make_move(action)

        if env.num_halfturns >= config.eval.max_game_length:
            env.choose_victor()

    if env.victor == Victor.draw:
        nextgen_score = 0.5
    elif (env.victor == Victor.white) == current_white:
        nextgen_score = 0
    else:
        nextgen_score = 1

    cur.append(cur_pipes)
    ng.append(ng_pipes)
    return nextgen_score, env, current_white

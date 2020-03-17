import chess
from deep_learning.config import Config, PlayWithHumanConfig
from deep_learning.workers import self_player, trainer, evaluate, supervised
import argparse
from deep_learning import lichess_game
import multiprocessing as mp
import sys
from logging import StreamHandler, basicConfig, DEBUG, getLogger, Formatter
import logging
def setup_logger(log_filename):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def create_arg_parser():
    """
        Command line arguments:
            sp: self play
            lip: accept a lichess challenger and play them
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="which command to do", choices=["sp", "lic", "tr", "eval", 'sl'])
    return parser
if __name__=="__main__":
    parser = create_arg_parser()
    arg = parser.parse_args()

    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)

    config = Config()
    setup_logger(config.resource.main_log_path)
    config.resource.create_directories()

    if arg.cmd == "lic":
        lichess_game.start(config, mode=arg.cmd)
    elif arg.cmd == "sp":
        self_player.start(config)
    elif arg.cmd == "tr":
        trainer.start(config)
    elif arg.cmd =='eval':
        evaluate.start(config)
    elif arg.cmd == 'sl':
        supervised.start(config)

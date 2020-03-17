import os

import os
import json
from datetime import datetime
from glob import glob
from logging import getLogger
from deep_learning.config import Resources

import chess
import pyperclip

logger = getLogger(__name__)

def pretty_print(env, chess_colors):
    new_pgn = open("test3.pgn", "at")
    game = chess.pgn.Game.from_board(env.board)
    game.headers["Result"] = env.result
    game.headers["White"], game.headers["Black"] = chess_colors
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    new_pgn.write(str(game) + "\n\n")
    new_pgn.close()
    pyperclip.copy(env.get_fen())

def write_game_data_to_file(path, data):
    try:
        with open(path, "wt") as f:
            json.dump(data, f)
    except Exception as e:
        print(e)

def get_game_data_filepaths(resource: Resources):
    general_filepath = os.path.join(resource.play_data_dir, resource.play_data_filename_tmpl % "*")
    filepaths = list(sorted(glob(general_filepath)))
    return filepaths

def get_next_generation_model_dirs(resource: Resources):
    directory_pat = os.path.join(resource.next_generation_model_dir, resource.next_generation_model_dirname_tmpl % "*")
    directories = list(sorted(glob(directory_pat)))
    return directories

def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)

def find_pgn_files(directory, pattern='*.pgn'):
    dir_pattern = os.path.join(directory, pattern)
    files = list(sorted(glob(dir_pattern)))
    return files

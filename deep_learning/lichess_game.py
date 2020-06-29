import li

from deep_learning.agent.model import ChessModel
from deep_learning.agent.player import Player
from deep_learning import lib
from deep_learning.config import Config, PlayWithHumanConfig
from deep_learning.environment import Chess, Victor

from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, ReadTimeout
from urllib3.exceptions import ProtocolError
from http.client import BadStatusLine

from multiprocessing import Manager, Pool, Process, Lock
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import logging

import chess

import json

logger = logging.getLogger(__name__)

def start(config: Config, mode: str):
    PlayWithHumanConfig().update_play_config(config.player_conf)

    return LichessPlayer(config, mode).start()

USER_NAME = "ChessterZero"
class LichessPlayer:

    def __init__(self, config:Config, mode: str):
        self.config = config
        self.liconf = self.config.liconf
        self.model = self.load_best_model()
        self.m = Manager()
        self.cur_pipes = self.model.get_pipes(self.config.player_conf.search_threads)
        self.mode = mode
        self.settings = self.load_settings(config.resource.lichess_settings_path)
        self.lichessAPI = li.API("1.1.4", self.settings['token'], "https://lichess.org", USER_NAME)

    def start(self):
        if self.mode == 'lic':
            self.play_lichess_challengers()

    def load_best_model(self):
        model = ChessModel(self.config)
        model.load(model.config.resource.model_best_conf_path, model.config.resource.model_best_weight_path)
        return model

    def load_settings(self, path):
        with open(path, 'r') as f:
            settings = json.load(f)
        return settings

    def play_lichess_challengers(self):

        print(self.settings)
        while True:

            event = self.event_stream()

            if event is None:
                continue
            elif event["type"] == "challenge":
                self.accept_challenge(event)
            elif event['type'] == "gameStart":
                play_game(event['game']['id'], self.lichessAPI, config=self.config, cur=self.cur_pipes)

    def event_stream(self):
        """
        Streams for events and returns the event found
        returns: (dict) event: the json event response from lichess. None if no response.
        """
        eventResponse = self.lichessAPI.eventStream()
        lines = eventResponse.iter_lines()
        line = next(lines)
        if line: # check for content in first event returned
            event = json.loads(line.decode('utf-8'))

            return event


    def accept_challenge(self, event):
        try: # this is racy..
            logger.debug("Challenge accepted from: {}".format(event['challenge']['challenger']['name']))
            self.lichessAPI.acceptChallenge(event['challenge']['id'])
        except:
            pass

def play_game(game_id, lichessAPI, config: Config, cur=None, event_queue=None):
    """
        Plays the game of chess against the user.  If the game falls behind.. i.e. the model is restarted
        the game with auto catch up.
        arguments:
            (string) game_id: the lichess game id
            (LichessAPI) lichessAPI: the lichessAPI calls object
            (Config) config: the config to play with
            (list(Connection)) the list of pipes used to communicate with the model
    """
    player = Player(config, pipes=cur)
    game_stream = lichessAPI.gameStream(game_id)
    lines = game_stream.iter_lines()
    line = next(lines)
    if line:
        game_json = json.loads(line.decode('utf-8'))

        initial_moves = game_json["state"]["moves"].split(' ')
        initial_moves = [] if initial_moves[0] == '' else initial_moves # detect empty board upon start
        env = setup_environment(initial_moves, game_json, game_id, player)
        if is_engines_turn(initial_moves, game_json) and not env.over():

            logger.debug("simulating")
            action = player.action(env)
            logger.debug(f"Making Initial Move: {action}")
            response = lichessAPI.makeMove(game_id, action)
            env.make_move(action) # catch up the board to lichess


    while not env.over():
        try:
            binary_chunk = next(lines)
        except(StopIteration):
            logger.debug("Reached end of stream")
            break
        try:
            gev_json = json.loads(binary_chunk.decode('utf-8')) if binary_chunk else None
            gev_type = gev_json["type"] if gev_json else None

            if gev_type == "gameState":
                moves = gev_json['moves'].split(' ')

                if is_engines_turn(moves, game_json):
                    env.make_move(moves[-1]) # catch up the board to lichess
                    env.print_pretty()

                    if not env.over(): # check for game end.
                        logger.debug("simulating")
                        action = player.action(env)
                        logger.debug(f"New Move: {action}")
                        response = lichessAPI.makeMove(game_id, action)
                        env.make_move(action) # update the board with pushed lichess move


            elif gev_type == "chatLine":
                print(gev_json)

        except (HTTPError, ReadTimeout, BadStatusLine, ChunkedEncodingError, ConnectionError, ProtocolError) as e:
            logger.debug(f"error detected {e}")
            current_games = lichessAPI.gamesPlaying()
            game_over = True # continue game upon weird exceptions from disconnections
            for game in current_games:
                logger.debug(f"Game: {game} | game_id: {game_id}")
                if game == game_id:
                    game_over = False
                    break

            if not game_over:
                continue
            else:
                break
    logger.debug("Game has ended.")
    if event_queue is not None:
        event_queue.put_nowait({"type" : "local_game_done"})

def is_engines_turn(moves, game_json):
    """
    Determines if it the engines turn to move
    Arguments:
        (list) moves: a list of uci chess moves
        (dict) game_json: the game's json 'gameFull' response from lichess's API
    return: whether or not it is the engines turn to move.
    """
    return (game_json["white"]["id"] == USER_NAME.lower() and len(moves) % 2 == 0) \
            or (game_json["black"]["id"] == USER_NAME.lower() and len(moves) % 2 == 1)

def setup_environment(list_of_moves, game_json, game_id, player):
    """
    Catches up the board to the current state
    return: (Chess) env: the board environment
    """
    env = Chess().reinit()

    logger.debug("Board history lost. Regenerating from: {}".format(list_of_moves))
    for move in list_of_moves:
        env.make_move(move)

    logger.debug("Board created :)")
    env.print_pretty()

    return env


# def seek(seek):
        # if seek == True:
        #     seekResponse = lichessAPI.seekChallenge(settings['gamesToSeek'])
        #     line = next(seekResponse.iter_lines())
        #     if line:
        #         seekEvent = json.loads(line.decode('utf-8'))
        #         print("Seek returns: {}".format(seekEvent))

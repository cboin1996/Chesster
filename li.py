"""
Lichess api interactions go here
"""

import json
import requests

import oauthlib
import chess
import requests

api_urls = {
    "profile": "/api/account",
    "playing": "/api/account/playing",
    "stream": "/api/bot/game/stream/{}",
    "stream_event": "/api/stream/event",
    "game": "/api/bot/game/{}",
    "move": "/api/bot/game/{}/move/{}",
    "chat": "/api/bot/game/{}/chat",
    "abort": "/api/bot/game/{}/abort",
    "accept": "/api/challenge/{}/accept",
    "decline": "/api/challenge/{}/decline",
    "upgrade": "/api/bot/account/upgrade",
    "resign": "/api/bot/game/{}/resign",
    "seek" : "/api/board/seek"
}
"""
Class is a wrapper for the lichess api
"""
class API:
    """
    Constructor
    args:
        version: lichess version
        token: the api token
        baseURL: the base api url
        userName: lichess bot account name
    """
    def __init__(self, version, token, baseURL, userName):
        self.token = token
        self.version = version
        self.baseURL = baseURL
        self.session = requests.Session()
        self.headers = {
                           'Authorization': 'Bearer {}'.format(token),
                           'User-Agent'   : "lichess-bot/{} user:{}".format(version, userName)
                         }
        self.session.headers.update(self.headers)


    def gamesPlaying(self, stream=False):
        """
        Find the current games being played
        args:
            stream: optional parameter to stream the request
        Returns: response
        """
        url = self.baseURL+api_urls["playing"]
        resp = self.session.get(url, stream=stream)
        return resp

    def resign(self, gameId: str):
        url = self.baseURL+api_urls["resign"]
        resp = self.session.post(url, data={"gameId":gameId})
        return resp

    def makeMove(self, gameId, move, data=None):
        """
        Makes a chess move
        args:
            gameId: lichess game id
            move: move to make
            data: data to post if needed
        Returns: response
        """
        url = self.baseURL+api_urls['move'].format(gameId, move)
        resp = self.session.post(url, data=data)
        return resp

    def eventStream(self):
        """
        Opens an event stream
        Returns: response
        """
        url = self.baseURL + api_urls['stream_event']
        return requests.get(url, headers=self.headers, stream=True)

    def acceptChallenge(self, challengeId):
        """
        Accepts a challenge from a user
        args:
            challengeId: challenger id given by lichess
        Returns: response
        """
        url = self.baseURL + api_urls['accept'].format(challengeId)
        return self.session.post(url)

    def gameStream(self, game_id):
        url = self.baseURL + api_urls['stream'].format(game_id)
        return requests.get(url, headers=self.headers, stream=True)

    def seekChallenge(self, gameParams):
        """
        (Broken) Seeks new challenges on lichess
        args:
            gameParams: seek parameters defined by user
        Returns: request response
        """
        url = self.baseURL + api_urls['seek']
        seekHeaders = dict(self.headers) # create new copy of the headers dict for this post only
        seekHeaders['Content-Type'] = "application/x-www-form-urlencoded"
        return requests.post(url, headers=seekHeaders, data=gameParams)

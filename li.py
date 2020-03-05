"""
Lichess api interactions go here
"""

import json
import requests

import oauthlib
import chess
import requests
# import lichess.api
# from lichess.format import PGN, PYCHESS

"""
Authenticates to lichess server
args: username of bot, oAUTH token
Return:
"""
# def getGames(userName, token):
#
#     return i.user_games(userName, max=100, auth=token, format=PYCHESS)
#

"""
Login to lichess
"""

urlDICT = {
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
    "resign": "/api/bot/game/{}/resign"
}

def makeApiCall(url, stream, session=None):
    header = { 'Authorization': 'Bearer {}'.format(devTok)}

    header["User-Agent"]: "lichess-bot/{} user:{}".format("1.1.4", "ChessterZero")

    session = requests.Session()
    session.headers.update(header)
    botAPI = "api/stream/event"
    resp = session.get(baseURL + botAPI, stream=True)
    print ( resp.text)

token = "szcWHpIqSpCc5cdC"
devTok = "NRFx8FGKuPKruR1Y"
authURL = "https://oauth.lichess.org/oauth/authorize"
tokURL = "https://oauth.lichess.org/oauth"

baseURL = "https://lichess.org/"
# streamURL = "https://lichess.org/api/bot/game/stream/{}".format(gameID)
# moveURL = "https://lichess.org/api/bot/game/{}/move/{}".format(gameID, move)


requests.get("https://lichess.org/api/account/")
makeApiCall(baseURL+urlDICT["playing"], False)

# header = { 'Authorization': 'Bearer {}'.format(devTok)}
# header["User-Agent"]: "lichess-bot/{} user:{}".format("1.1.4", "ChessterZero")

# session = requests.Session()
# session.headers.update(header)
# botAPI = "api/stream/event"
# resp = session.get(baseURL + botAPI, stream=True)
# print ( resp.text)

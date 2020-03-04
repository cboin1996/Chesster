"""
Lichess api interactions go here
"""

import chess
import requests
import lichess.api
from lichess.format import PGN, PYCHESS

"""
Authenticates to lichess server
args: username of bot, oAUTH token
Return:
"""
def getGames(userName, token):
    
    return lichess.api.user_games(userName, max=100, auth=token, format=PYCHESS)

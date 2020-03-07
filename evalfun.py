import math
import random
import itertools
import os
import chess

def start(config: Config):                 
    return EvaluateWorker(config).start() 
"""
^ creates a new function to start the evaluation function. 
This "Config" is from the utils package which may need to be downloaded
using pip install. If this doesn't work, will probably sub with 
configparser instead. 
"""
class evalFunc(someGameState):
    def intialize(agent, config: Config): 
        self.config = config
        self.play_config = config.eval.play_config
        self.current
    
    

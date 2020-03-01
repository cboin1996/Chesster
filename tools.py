"""
Chess Tools such as evaluating position/winning and losing go here
"""

import chess


"""
Evaluate Position Function to determine the score of the position
arg: Chess board

Return: Number
"""
def evaluatePosition(board):
    valueDict = {'P':1, 'p': -1, 'N': 3, 'n': -3, 'B': 3, 'b': -3,
                 'R': 5, 'r': -5, 'Q': 9, 'q': -9, 'K': 10000, 'k':-10000}

    pieceList = str(board).replace("/","").replace("\n"," ").replace(". ", "").split(" ")
    print(pieceList)
    pieceValues = [valueDict[pieceChar] for pieceChar in pieceList]
    print(sum(pieceValues))

"""
Checks if game ends
"""
def gameEnd(board):
    if(board.is_game_over() or board.is_sufficient_material() or board.can_claim_threefold_repetition()):
        return -1
    if(board.is_game_over()):
        return 1

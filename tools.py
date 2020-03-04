"""
Chess Tools such as evaluating position/winning and losing go here
"""

import logging
import chess

STRONGEST_INITIAL = 9999

"""
Evaluate Position Function to determine the score of the position
arg: Chess board

Return: value of board
"""
def evaluatePosition(board):
    valueDict = {'P':1, 'p': -1, 'N': 3, 'n': -3, 'B': 3, 'b': -3,
                 'R': 5, 'r': -5, 'Q': 9, 'q': -9, 'K': 10000, 'k':-10000, '.': 0}

    pieceList = str(board).replace("/","").replace("\n"," ").replace(". ", "").split(" ")
    pieceValues = [valueDict[pieceChar] for pieceChar in pieceList]
    return sum(pieceValues)

"""
Checks if game ends
"""
def gameEnd(board):
    if(board.is_sufficient_material() or board.can_claim_threefold_repetition()):
        logging.info("Draw Occured")
        return -1
    if(board.is_game_over()):
        logging.info("Checkmate Occured")
        return 1

"""
Initiate minimax search

Return: move to perform
"""
def startMinimax(board, depth, colour):
    legalMoves = board.legal_moves
    strongestMove = -STRONGEST_INITIAL
    strongestFinalMove = None

    for move in legalMoves:
        move = chess.Move.from_uci(str(move))
        logging.debug("Checking move: {}".format(move))
        board.push(move)
        minimaxValue = max(strongestMove, minimax(board, depth-1, -(STRONGEST_INITIAL+1), STRONGEST_INITIAL+1, not colour))
        logging.debug("Move: {} produced a minimax value of: {}".format(move, minimaxValue))

        board.pop()
        if minimaxValue > strongestMove:
            logging.debug("Minimax value ({}) is greater than strongest move ({})".format(minimaxValue, strongestMove))
            strongestMove = minimaxValue
            strongestFinalMove = move
            logging.info("Current optimal move is {}, it has a Minimax Value of {}".format(str(strongestFinalMove), minimaxValue))
        else:
            logging.debug("Minimax value ({}) is less than strongest move ({})".format(minimaxValue, strongestMove))
    return strongestFinalMove
"""
Recursive minimax search to find best move

Args:

board: Python-Chess board objct
depth: depth of search ex: 3 = 3 moves deep
alpha: Alpha value of minimax search
beta: Beta value of minimax search
colour: Colour of search, white = True, black = false

Return: best move
"""

def minimax(board, depth, alpha, beta, colour):
    logging.debug("Performing minimax with alpha: {} and beta {} on colour {}".format(alpha, beta, colour))
    if depth == 0: # base case for recusion
        return -evaluatePosition(board)
    legalMoves = board.legal_moves
    if colour:
        strongestMove = -STRONGEST_INITIAL
    else:
        strongestMove = STRONGEST_INITIAL

    for move in legalMoves:
        move = chess.Move.from_uci(str(move))
        board.push(move)
        if colour:
            strongestMove = max(strongestMove, minimax(board, depth-1, alpha, beta, not colour))
            alpha = max(alpha, strongestMove)
        else:
            strongestMove = min(strongestMove, minimax(board, depth-1, alpha, beta, not colour))
            beta = min(beta, strongestMove)

        board.pop()
        if beta <= alpha:
            return strongestMove
    return strongestMove

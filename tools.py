"""
Chess Tools such as evaluating position/winning and losing go here
"""

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
    if(board.is_game_over() or board.is_sufficient_material() or board.can_claim_threefold_repetition()):
        return -1
    if(board.is_game_over()):
        return 1

"""
Initiate minimax search

Return: move to perform
"""
def startMinimax(board, depth, minimizing, color):
    legalMoves = board.legal_moves
    strongestMove = -STRONGEST_INITIAL
    strongestFinalMove = None

    for move in legalMoves:
        move = chess.Move.from_uci(str(move))
        board.push(move)
        minimaxValue = max(strongestMove,
                           minimax(board,
                                   depth-1,
                                   -(STRONGEST_INITIAL+1),
                                   STRONGEST_INITIAL+1,
                                   not minimizing,
                                   color))
        board.pop()
        if minimaxValue > strongestMove:
            print("Minimax Value: ", minimaxValue)
            strongestMove = minimaxValue
            strongestFinalMove = move
            print("Optimal Score: ", str(strongestMove))
            print("Optimal move: ", str(strongestFinalMove))
    return strongestFinalMove
"""
Recursive minimax search to find best move
args:
    board: chess board object
    depth: depth of the minimax algorithm
    alpha: alpha param
    beta: beta param
    minimizing: max or minimized search
    colour: chess colour True if white 
Return: best move
"""

def minimax(board, depth, alpha, beta, minimizing, color):
    if depth == 0: # base case for recusion
        if color == True:
            return evaluatePosition(board)
        else:
            return -evaluatePosition(board)
    legalMoves = board.legal_moves
    if minimizing:
        strongestMove = -STRONGEST_INITIAL
    else:
        strongestMove = STRONGEST_INITIAL

    for move in legalMoves:
        move = chess.Move.from_uci(str(move))
        board.push(move)
        if minimizing:
            strongestMove = max(strongestMove, minimax(board, depth-1, alpha, beta, not minimizing, color))
            alpha = max(alpha, strongestMove)
        else:
            strongestMove = min(strongestMove, minimax(board, depth-1, alpha, beta, not minimizing, color))
            beta = min(beta, strongestMove)

        board.pop()
        if beta <= alpha:
            return strongestMove
    return strongestMove

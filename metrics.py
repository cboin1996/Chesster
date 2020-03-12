import chess
from tools import boardToArray

def evalKingSafety(board):
    boardToArray(board)


def evalTotalPieceMobility(board):
    pass

def evalAttackingEnemyKing(board):
    pass

def evalControlCenterofBoard(board):
    pass

def evalConnectedPawns(board):
    pass

def evalDoubledPawns(board):
    pass

def proximityToPromotion(board):
    pass

def evalIsolatedPawns(board):
    pass

def evalPassed(board):
    pass


def evalBishopPair(board):
    pass

def evalConnectedRooks(board):
    pass

def abilityToCastle(board):
    pass

if __name__ == "__main__":
    board = chess.Board()
    evalKingSafety(board)

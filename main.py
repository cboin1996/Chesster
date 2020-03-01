"""
Main starting point for the chess program
"""
import tools
import chess
import sys


"""
Runs the main chess program

Return: None
"""
def playChess():
    board = chess.Board()
    while True:
        print(board.unicode(invert_color=True))
        flag = False
        while not flag:
            move = input("Make a move (e2e4): ")
            try:
                flag = chess.Move.from_uci(move) in board.legal_moves
            except:
                print("Invalid input. Try Again.")
        # TODO: check for valid move
        move = chess.Move.from_uci(str(move))
        board.push(move)
        print(board.unicode(invert_color=True))
        computerMove = tools.startMinimax(board, 4, True)
        computerMove = chess.Move.from_uci(str(computerMove))
        board.push(computerMove)

if __name__=="__main__":

    playChess()

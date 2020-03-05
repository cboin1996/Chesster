"""
Main starting point for the chess program
"""
import tools
import chess
import sys
import logging


"""
Runs the main chess program
depth: the depth of the AI's alpha-beta search
Return: None
"""
def playChess(depth):
    board = chess.Board()
    while True:
        print(board.unicode(invert_color=True))
        flag = False
        while not flag:
            move = input("Make a move (in the form e2e4): ")
            try:
                flag = chess.Move.from_uci(move) in board.legal_moves
            except:
                logging.debug("{} is not a valid move".format(move))
        # TODO: check for valid move
        move = chess.Move.from_uci(str(move))
        board.push(move)
        print(board.unicode(invert_color=True))
        computerMove = tools.startMinimax(board, depth, True)
        computerMove = chess.Move.from_uci(str(computerMove))
        board.push(computerMove)

"""
Get the depth of the AI, and run program with that depth
"""

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    flag = False
    while not flag:
        try:
            if len(sys.argv) > 1:
                depth = int(sys.argv[1])
                break
            else:
                depth = int(input("Enter the depth for your AI: "))
                flag = True
        except KeyboardInterrupt:
            raise
        except ValueError:
            print("Integer input for depth required.")


    playChess(depth)

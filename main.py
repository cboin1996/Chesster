"""
Main starting point for the chess program
"""
import tools
import chess
import sys, os, json
import li

"""
Runs the main chess program
arg: the depth of the AI's alpha-beta search
Return: None
"""
def playChess(depth):
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
        computerMove = tools.startMinimax(board, depth, True)
        computerMove = chess.Move.from_uci(str(computerMove))
        board.push(computerMove)
"""
Runs the chess program using lichess
arg: the depth of the AI's alpha-beta search
Return: None
"""
def playLichess(depth):
    with open(os.path.join(sys.path[0], 'secret.json'), 'r') as f:
        settings = json.load(f)
    print(settings)

    while True:
        games = li.getGames('ChessterZero', settings['token'])
        print(next(games).end().board())
"""
Get the depth of the AI, and run program with that depth
"""

if __name__=="__main__":
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

    mode = input("Lichess or terminal (l/t): ")
    if mode == 'l':
        playLichess(depth)
    elif mode == 't':
        playChess(depth)
    else:
        print("You didn't follow my instructions so i quit as cb is too lazy to error catch rn.")

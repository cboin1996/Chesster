"""
Main starting point for the chess program
"""
import tools
import chess
import sys, os, json
import li
import logging

"""
Runs the main chess program
args:
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
        computerMove = tools.startMinimax(board, depth, True, False)
        computerMove = chess.Move.from_uci(str(computerMove))
        board.push(computerMove)
"""
Runs the chess program using lichess
arg:
    depth: the depth of the AI's alpha-beta search
    seek: automatically find a game
Return: None
"""
def playLichess(depth, seek=False):
    with open(os.path.join(sys.path[0], 'secret.json'), 'r') as f:
        settings = json.load(f)
    print(settings)

    lichessAPI = li.API("1.1.4", settings['token'], "https://lichess.org", "ChessterZero")

    while True:
        # check for new challenges
        eventResponse = lichessAPI.eventStream()
        if seek == True:
            seekResponse = lichessAPI.seekChallenge(settings['gamesToSeek'])
            line = next(seekResponse.iter_lines())
            if line:
                seekEvent = json.loads(line.decode('utf-8'))
                print("Seek returns: {}".format(seekEvent))

        lines = eventResponse.iter_lines()
        line = next(lines)
        if line: # check for content in first event returned
            event = json.loads(line.decode('utf-8'))

        try:
            print("Challenge accepted from: {}".format(event['challenge']['challenger']['name']))
            lichessAPI.acceptChallenge(event['challenge']['id'])
        except:
            pass

        response = lichessAPI.gamesPlaying().json()

        for game in response['nowPlaying']:
            board = chess.Board(fen=game['fen'])
            if game['color'] == 'black':
                board.turn = chess.BLACK
                color = False
            else:
                board.turn = chess.WHITE
                color = True

            if game['isMyTurn'] == True:
                print('') # escapes the no newline prints from before
                move = tools.startMinimax(board, depth, True, color)
                if str(move) == "No":
                    for move in board.legal_moves:
                        break
                move = chess.Move.from_uci(str(move))
                response = lichessAPI.makeMove(game['fullId'], move)
            else:
                print('\rWaiting for my turn... ', end='')

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

    mode = input("Lichess or terminal (l/t): ")
    if mode == 'l':
        playLichess(depth)
    if mode == 'ls':
        playLichess(depth, True)
    elif mode == 't':
        playChess(depth)
    else:
        print("You didn't follow my instructions so i quit as cb is too lazy to error catch rn.")

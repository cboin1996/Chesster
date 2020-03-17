"""
Wrapper for the python-chess module
"""

import chess
import numpy as np
import enum
import copy
import chess.pgn



Victor = enum.Enum("Victor", "black white draw")
class Chess:
    """
        Represents a chess environ
        attributes:
            board: chess board
            result: result from python "0-1" etc.
            victor: winner colour str
    """
    def __init__(self):
        self.board = None
        self.result = None
        self.victor = None
        self.isResigned = False
        self.num_halfturns = 0

    def reinit(self):
        self.board = chess.Board()
        self.victor = None
        self.isResigned = False
        self.num_halfturns = 0
        return self

    def match_fen(self, fen:str):
        self.board = chess.Board(fen=fen)
        self.victor = None
        self.isResigned = False

    def over(self):
        # checks for game over
        return self.victor is not None

    def blacks_turn(self):
        return self.board.turn == chess.BLACK

    def whites_turn(self):
        return self.board.turn == chess.WHITE

    def print_pretty(self):
        print(self.board.unicode(invert_color=True)) # prints upside down otherwise

    def print(self):
        print(self.board)

    def legal_moves(self):
        return self.board.legal_moves

    def get_fen(self):
        return self.board.fen()

    def canon_input_planes(self):
        """
            Gets a (18,8,8) representation of the board state for feeding into the network
        """
        return canon_input_planes(self.get_fen())

    def choose_victor(self):
        """
        Determines a victor based on material advantage
        """
        score = self.testeval(absolute=True)
        if abs(score) < 0.01:
            self.victor = Victor.draw
            self.result = "1/2-1/2"
        elif score > 0:
            self.victor = Victor.white
            self.result = "1-0"
        else:
            self.victor = Victor.black
            self.result = "0-1"


    def make_move(self, action: str, check_finished = True):
        """
            Takes an action from the policy and makes the move
            args:
                action: game move uci string
                check_finished: check if game is done or not

        """
        if action is None and check_finished:
            self._resign()
            return
        move = chess.Move.from_uci(action)
        self.board.push(move)
        self.num_halfturns += 1

        if check_finished and self.board.result(claim_draw=True) != '*':
            self._game_over()

    """
        Returns a copy the chess environment
    """
    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env


    """
        Private methods for determining if game is over or if player resigned
    """
    def _resign(self):
        self.isResigned = True

        if self.whites_turn():
            self.result = '0-1'
            self.victor= Victor.black
        elif self.blacks_turn():
            self.result = '1-0'
            self.victor = Victor.white

    def _game_over(self):
        if self.victor is None:
            self.result = self.board.result(claim_draw=True) # claim_draw checks for three fold rep etc.

            if self.result == "1-0":
                self.victor = Victor.white

            elif self.result == "0-1":
                self.victor = Victor.black

            else:
                self.victor = Victor.draw

    def testeval(self, absolute=False) -> float:
        return testeval(self.board.fen(), absolute)

"""
Input planes declarations
"""
pieces_order = 'KQRBNPkqrbnp' # 12x8x8
castling_order = 'KQkq'       # 4x8x8
# fifty-move-rule             # 1x8x8
# en en_passant               # 1x8x8

ind = {pieces_order[i]: i for i in range(12)}

def canon_input_planes(fen):
    """
    Arguments:
        fen: the chess board fen respresntation string
        return: (18,8,8) board representation
    """
    fen = maybe_flip_fen(fen, is_black_turn(fen))
    return all_input_planes(fen)


def all_input_planes(fen):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
    """
    current_aux_planes = aux_planes(fen)

    history_both = to_planes(fen)

    ret = np.vstack((history_both, current_aux_planes))
    assert ret.shape == (18, 8, 8)
    return ret


def maybe_flip_fen(fen, flip = False):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
    """
    if not flip:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])
    return "/".join([swapall(row) for row in reversed(rows)]) \
        + " " + ('w' if foo[1] == 'b' else 'b') \
        + " " + "".join(sorted(swapall(foo[2]))) \
        + " " + foo[3] + " " + foo[4] + " " + foo[5]


def aux_planes(fen):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
        Used to map a representation of enpassant rights, castle rights, and fifty move count
    """
    foo = fen.split(' ')

    en_passant = np.zeros((8, 8), dtype=np.float32)
    if foo[3] != '-':
        eps = alg_to_coord(foo[3])
        en_passant[eps[0]][eps[1]] = 1

    fifty_move_count = int(foo[4])
    fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

    castling = foo[2]
    auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
                        np.full((8, 8), int('Q' in castling), dtype=np.float32),
                        np.full((8, 8), int('k' in castling), dtype=np.float32),
                        np.full((8, 8), int('q' in castling), dtype=np.float32),
                        fifty_move,
                        en_passant]

    ret = np.asarray(auxiliary_planes, dtype=np.float32)
    assert ret.shape == (6, 8, 8)
    return ret

def to_planes(fen):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
        Used to map boards to individual planes for the pieces.
    """
    board_state = replace_tags_board(fen)
    pieces_both = np.zeros(shape=(12, 8, 8), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            v = board_state[rank * 8 + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (12, 8, 8)
    return pieces_both


def alg_to_coord(alg):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
    """
    rank = 8 - int(alg[1])        # 0-7
    file = ord(alg[0]) - ord('a') # 0-7
    return rank, file


def coord_to_alg(coord):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
    """
    letter = chr(ord('a') + coord[1])
    number = str(8 - coord[0])
    return letter + number

def replace_tags_board(board_san):
    """
        Method taken from https://https://github.com/Zeta36/chess-alpha-zero
    """
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    return board_san.replace("/", "")

def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'

def testeval(fen, absolute = False) -> float:
    piece_vals = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1} # somehow it doesn't know how to keep its queen
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue

        if c.isupper():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.upper()]
            tot += piece_vals[c.upper()]
    v = ans/tot
    if not absolute and is_black_turn(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3) # arbitrary

"""
Automatically plays the game using the Alpha Go Zero MCTS algorithm

Module is taken and adapted from https://github.com/Zeta36/chess-alpha-zero
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import chess
import numpy as np

from deep_learning.environment import Chess, Victor
from deep_learning.agent.model import ChessModel
from deep_learning.config import Config



"""
    attributes:
        q: mean action value (avg. game result across sims that take a)
        p: prior probs as fetched from the network
        n: visit count/num of times weve taken an action during sims
        w: every time a child of this action is visited by the algorithm,
            this accumulates the value (calculated from the value network) of that child. This is modified
            by a virtual loss which encourages threads to explore different nodes.

"""
class ActionStats:
    def __init__(self):
        self.q = 0
        self.p = 0
        self.n = 0
        self.w = 0
"""
    attributes:
        a: holds the action stats from above
        sum_n: sum of the visit count (n) from each attribute of ActionStats
"""
class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0
"""
Interacts with environment.py to play the game.
    moves: stores list of moves
    VisitStats tree: holds the visited game states and actions
    int labels: list of uci labels (e2e4, e2e5)
    int num_labels: number of labels
"""
class Player:

    def __init__(self, config: Config, pipes=None, play_conf=None, dummy=False): # add model here in init?
        self.moves = []
        self.tree = defaultdict(VisitStats)
        self.config = config
        self.play_conf = play_conf or self.config.player_conf
        self.labels = config.labels
        self.num_labels = config.num_labels
        self.move_lookup = {chess.Move.from_uci(move) : i for move, i in zip(self.labels, range(self.num_labels))}
        if dummy:
            return
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)

    def reset(self):
        """
        Reset the tree.
        """
        self.tree = defaultdict(VisitStats)


    def action(self, env, can_stop = True) -> str:
        """
        Figures out the next best move
        within the specified environment and returns a string describing the action to take.
        arguments:
            ChessEnv env: environment in which to figure out the action
            boolean can_stop: whether we are allowed to take no action (return None)
        returns: None if no action should be taken (indicating a resign). Otherwise, returns a string
            indicating the action to take in uci format
        """
        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.num_labels), p = self.apply_temperature(policy, env.num_halfturns)))

        if can_stop and self.play_conf.resign_threshold is not None and \
                        root_value <= self.play_conf.resign_threshold \
                        and env.num_halfturns > self.play_conf.min_resign_turn:
            # noinspection PyTypeChecker
            return None
        else:
            self.moves.append([env.get_fen(), list(policy)])
            return self.config.labels[my_action]

    def search_moves(self, env) -> (float, float):
        """
        Find the highest value move using MCTS.  Gets multiple estimates using threading so we can pick best.

        arguments:
            env: env to search for moves within
        return: (float,float): the maximum value of all values predicted by each thread,
            and the first value that was predicted.
        """
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_conf.search_threads) as executor:
            for _ in range(self.play_conf.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0] # vals[0] is kind of racy

    def search_my_move(self, env:Chess, is_root_node=False) -> float:
        """
            Q, V are the values for the player (which is always white)
            P is the value for the next player (black or white)

            Searches for possible moves, adds to the tree and returns the best move that is found from the search

            arguments:
                env: Chess environment
                is_root_node: whether were at the root node of the MCTS
                return: value of the move (from network prediction)
        """
        if env.over():
            if env.victor == Victor.draw:
                return 0
            # assert env.whitewon != env.white_to_move # side to move can't be winner!
            return -1

        state = board_state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_conf.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.make_move(action_t.uci())
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env: Chess) -> (np.ndarray, float):
        """
        Expand into a new leafm get a prediction for the policy and value of a state

        returns: the policy and value predic for the state
        """
        state_planes = env.canon_input_planes()

        leaf_p, leaf_v = self.predict(state_planes)

        if not env.whites_turn():
            leaf_p = Config.flip_policy(leaf_p)

        return leaf_p, leaf_v

    def predict(self, state_planes):

        """
        Gets prediction from policy value network
            arguments:
                state_planes: obs state represented as planes
            returns(float, float):
                policy -- the prior probability of taking the action leading here
                value: the value of the state prediction
        """

        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    def select_action_q_and_u(self, env, is_root_node) -> chess.Move:
        """
        Picks the next action to explore using the AGZ MCTS algorithm.

        Picks based on the action which maximizes the maximum action value
        (ActionStats.q) + an upper confidence bound on that action.
        arguments:
            Chess env: env to look for the next moves within
            is_root_node: whether this is for the root node of the MCTS search.
        returns: chess.Move: the move to explore
        """

        # this method is called with state locked
        state = board_state_key(env)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None: #push p to edges
            tot_p = 1e-8
            for mov in env.legal_moves():
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.play_conf.noise_eps
        c_puct = self.play_conf.c_puct
        dir_alpha = self.play_conf.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))

        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn):
        """
        Applies a random fluctuation to probability of choosing various actions
        arguments:
            policy: list of probabilities of taking each action
            turn: number of turns that have occurred in the game so far
        returns: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp
            = less.
        """
        tau = np.power(self.play_conf.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.num_labels)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        """calc π(a|s0)
        returns: list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = board_state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.num_labels)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def sl_action(self, observation, my_action, weight=1):
        """
        Logs the action in self.moves. Useful for generating a game using game data (supervised learning)
        Arguments:
            str observation: FEN format observation indicating the game state
            str my_action: uci format action to take
            float weight: weight to assign to the taken action when logging it in self.moves
        return str: the action, unmodified.
        """
        policy = np.zeros(self.num_labels)

        k = self.move_lookup[chess.Move.from_uci(my_action)]
        policy[k] = weight

        self.moves.append([observation, list(policy)])
        return my_action

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.

        arguments:
            self:
            z: win=1, lose=-1, draw=0

        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

"""
Returns the fen from a chess board
argumments:
    env: Chess environment
    return: fen string
"""
def board_state_key(env: Chess):
    fen = env.get_fen().rsplit(' ', 1) # no need for move clock
    return fen[0]

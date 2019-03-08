import Field
import Tile
import State
import random
import time
import math


class Tetris:

    # declare the game tiles
    I = Tile.Tile([[1, 1, 1, 1]], '\x1b[6;30;46m')
    J = Tile.Tile([[1, 0, 0], [1, 1, 1]], '\x1b[6;30;44m')
    L = Tile.Tile([[0, 0, 1], [1, 1, 1]], '\x1b[6;30;43m')
    O = Tile.Tile([[1, 1], [1, 1]], '\x1b[6;30;47m')
    S = Tile.Tile([[0, 1, 1], [1, 1, 0]], '\x1b[6;30;42m')
    T = Tile.Tile([[1, 0], [1, 1], [1, 0]], '\x1b[6;30;45m')
    Z = Tile.Tile([[1, 1, 0], [0, 1, 1]], '\x1b[6;30;41m')

    TILES = [I, J, L, O, S, T, Z]

    ROW_GAIN = 1

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.states = {}

        for i in range(0, len(Tetris.TILES)):
            field = Field.Field(n, m)
            self.states[(field.representation(), i)] = State.State(field, i, self.states)

    # single pass of the value-iteration algorithm
    def update(self):
        states = list(self.states.values())
        n_states = len(states)
        print("New iteration with %d states" % n_states)
        total_delta = 0
        for state in states:
            total_delta += state.mdp_update()
        return total_delta, n_states != len(self.states)

    # performs the value-iteration algorithm
    def optimize(self):
        growing = True
        delta = 0
        while growing or delta > 1e-6:
            delta, growing = self.update()
            print("Total delta of expectations : %f" % delta)

    # simulates one game and applies the n-step semi-gradient TD algorithm
    def episode(self, w, alpha, epsilon):
        score = 0
        T = 1000000
        tau = 0
        t = 0
        n = 20
        gamma = 0.99995

        states = [Field.Field(self.n, self.m)]
        rewards = [0]
        tiles = []

        while tau < T:
            if t < T:
                if len(tiles) == 0:
                    tiles = [i for i in range(len(self.TILES))]
                tile = tiles[random.randint(0, len(tiles) - 1)]
                tiles.remove(tile)
                move = State.State(states[t], tile).vf_train_move(w, epsilon)

                if move is None:
                    T = t + 1
                    rewards.append(-1000000000)
                    states.append(states[t])
                else:
                    (next_field, gain, game_gain) = states[t].successor(self.TILES[tile], move)

                    states.append(next_field)
                    rewards.append(gain)

                    score += game_gain

            tau = t - n + 1

            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+n, T)):
                    G += rewards[i] * (gamma ** (i - tau - 1))

                if tau + n < T:
                    G += states[tau + n].utility(w)[0] * (gamma ** n)

                w = states[tau].utility_update(w, alpha, G)

            t += 1

        return w, score

    # runs the value-function approximation algorithm
    def learn(self):
        w = [0 for i in range(Field.Field(self.n, self.m).dimension())]

        n_episodes = 50
        mod = n_episodes / 10

        scores = []
        sum = 0
        min_score =  1000000
        max_score = -1000000

        for k in range(n_episodes+1):
            w, score = self.episode(w, math.exp(-k), 1 / (1 + 16 * math.log(k+1)))
            scores.append(score)
            sum += score
            min_score = min(min_score, score)
            max_score = max(max_score, score)

            if k % mod == 0:
                print("Iteration %d  avg. score %f [%f, %f]" % (k, sum / mod, min_score, max_score))
                sum = 0
                min_score =  1000000
                max_score = -1000000

        print(w)
        self.w = w

    # compares the performances of several approaches
    def compare_perf(self):
        tests = 15
        #sum_mdp = 0
        sum_vf = 0
        sum_rnd = 0
        sum_low = 0
        sum_hol = 0

        for i in range(tests):
            #current_field_mdp = Field.Field(self.n, self.m)
            current_field_vf = Field.Field(self.n, self.m)
            current_field_rnd = Field.Field(self.n, self.m)
            current_field_low = Field.Field(self.n, self.m)
            current_field_hol = Field.Field(self.n, self.m)
            #score_mdp = 0
            score_vf = 0
            score_rnd = 0
            score_low = 0
            score_hol = 0
            #end_mdp = False
            end_vf = False
            end_rnd = False
            end_hol = False
            end_low = False

            tiles = []

            while not end_vf or not end_hol or not end_low or not end_rnd: # or not end_mdp
                if len(tiles) == 0:
                    tiles = [i for i in range(len(self.TILES))]
                tile = tiles[random.randint(0, len(tiles) - 1)]
                tiles.remove(tile)

                # if not end_mdp:
                #     move_mdp = self.states[(current_field_mdp.representation(), tile)].mdp_move()
                #     if move_mdp is None:
                #         end_mdp = True
                #     else:
                #         (current_field_mdp, my_gain, gain) = current_field_mdp.successor(self.TILES[tile], move_mdp)
                #         score_mdp += gain

                if not end_vf:
                    move_vf = State.State(current_field_vf, tile).vf_move(self.w)
                    if move_vf is None:
                        end_vf = True
                    else:
                        (current_field_vf, my_gain, gain) = current_field_vf.successor(self.TILES[tile], move_vf)
                        score_vf += gain

                if not end_rnd:
                    move_rnd = State.State(current_field_rnd, tile).random_move()
                    if move_rnd is None:
                        end_rnd = True
                    else:
                        (current_field_rnd, my_gain, gain) = current_field_rnd.successor(self.TILES[tile], move_rnd)
                        score_rnd += gain

                if not end_hol:
                    move_hol = State.State(current_field_hol, tile).hole_move()
                    if move_hol is None:
                        end_hol = True
                    else:
                        (current_field_hol, my_gain, gain) = current_field_hol.successor(self.TILES[tile], move_hol)
                        score_hol += gain

                if not end_low:
                    move_low = State.State(current_field_low, tile).lowest_move()
                    if move_low is None:
                        end_low = True
                    else:
                        (current_field_low, my_gain, gain) = current_field_low.successor(self.TILES[tile], move_low)
                        score_low += gain

            #sum_mdp += score_mdp
            sum_vf += score_vf
            sum_hol += score_hol
            sum_rnd += score_rnd
            sum_low += score_low

            print(str(i+1) + " done")

        return sum_vf/tests, sum_rnd/tests, sum_low/tests, sum_hol/tests  # , sum_mdp/tests

    # tests the performances of the value-function approximation algorithm
    def test_vf(self, w):
        tests = 25
        sum_vf = 0

        for i in range(tests):
            current_field_vf = Field.Field(self.n, self.m)
            score_vf = 0
            end_vf = False

            tiles = []

            while not end_vf:
                if len(tiles) == 0:
                    tiles = [i for i in range(len(self.TILES))]
                tile = tiles[random.randint(0, len(tiles) - 1)]
                tiles.remove(tile)

                if not end_vf:
                    move_vf = State.State(current_field_vf, tile).vf_move(w)
                    if move_vf is None:
                        end_vf = True
                    else:
                        (current_field_vf, my_gain, gain) = current_field_vf.successor(self.TILES[tile], move_vf)
                        score_vf += gain

            sum_vf += score_vf
            print(str(i+1) + " done")

        return sum_vf / tests

    # play the game with a specific algorithm
    # opt :
    #   0 value-iteration (run optimize() before)
    #   1 value-function approximation (run learn() before)
    #   2 random
    #   3 lowest move
    #   4 minimum number of holes
    def play(self, opt=1):
        current_field = Field.Field(self.n, self.m)
        score = 0
        my_gain = 0

        tiles = []

        while True:
            if len(tiles) == 0:
                tiles = [i for i in range(len(self.TILES))]
            tile = tiles[random.randint(0, len(tiles)-1)]
            tiles.remove(tile)

            print("Current score : %d" % score)
            print("Current board :")
            current_field.print()
            print("Tile to place :")
            self.TILES[tile].print()

            if opt == 0:
                move = self.states[(current_field.representation(), tile)].mdp_move()
            elif opt == 1:
                move = State.State(current_field, tile).vf_move(self.w)
            elif opt == 2:
                move = State.State(current_field, tile).random_move()
            elif opt == 3:
                move = State.State(current_field, tile).lowest_move()
            elif opt == 4:
                move = State.State(current_field, tile).hole_move()

            if move is None:
                print("Impossible to place the tile !")
                break

            (current_field, my_gain, gain) = current_field.successor(self.TILES[tile], move)
            score += gain

            time.sleep(2)

        print("GAME OVER ! Score : %d" % score)
        current_field.print()

    def print(self):
        for state in self.states.values():
            state.print()


if __name__ == '__main__':
    game = Tetris(20, 10)
    game.learn()

    play = True
    while play:
        game.play()
        play = input('Play again ? (y/n) : ') == 'y'

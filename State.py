import random


class State:

    def __init__(self, field, tile, states=None):
        self.states = states
        self.field = field
        self.tile = tile
        self.expectation = 0.0
        self.decision = -1
        self.moves = []
        self.next_fields = []
        self.gains = []
        self.true_gains = []

        import Tetris
        for move in field.positions(Tetris.Tetris.TILES[tile]):
            self.moves.append(move)
            (successor, gain, true_gain) = field.successor(Tetris.Tetris.TILES[tile], move)
            self.next_fields.append(successor)
            self.gains.append(gain)
            self.true_gains.append(true_gain)

    # updates this state in the value-iteration algorithm
    def mdp_update(self):
        import Tetris
        new_expectation = -1.0
        p = 1.0 / len(Tetris.Tetris.TILES)

        for i in range(0, len(self.moves)):
            move_expectation = self.true_gains[i]

            for next_tile in range(0, len(Tetris.Tetris.TILES)):
                if (self.next_fields[i].representation(), next_tile) not in self.states:
                    self.states[(self.next_fields[i].representation(), next_tile)] = State(self.next_fields[i],
                                                                                           next_tile, self.states)
                successor = self.states[(self.next_fields[i].representation(), next_tile)]
                move_expectation += p * successor.expectation

            if move_expectation > new_expectation:
                new_expectation = move_expectation
                self.decision = i

        delta = abs(self.expectation - new_expectation)
        self.expectation = new_expectation
        return delta

    # returns the best move according to the value-iteration policy
    def mdp_move(self):
        if self.decision == -1:
            return None
        return self.moves[self.decision]

    def vf_move(self, w):
        move = None
        max_value = 0

        for i in range(len(self.moves)):
            if move is None or self.next_fields[i].utility(w)[0] + self.gains[i] > max_value:
                max_value = self.next_fields[i].utility(w)[0] + self.gains[i]
                move = self.moves[i]

        return move

    # returns the best move according to the value-function approximation policy
    def vf_train_move(self, w, epsilon):
        if len(self.moves) == 0:
            return None

        if random.random() > epsilon:
            return self.vf_move(w)

        return self.moves[random.randint(0, len(self.moves)-1)]

    # returns the best move according to the lowest move policy
    def lowest_move(self):
        if len(self.moves) == 0:
            return None

        minheight = -1
        index = -1
        for i in range(len(self.moves)):
            if self.moves[i][0] > minheight:
                minheight = self.moves[i][0]
                index = i

        return self.moves[index]

    # returns a random move
    def random_move(self):
        if len(self.moves) == 0:
            return None

        return self.moves[random.randint(0, len(self.moves)-1)]

    # returns the move leading to the minimum number of holes
    def hole_move(self):
        if len(self.moves) == 0:
            return None

        minholes = 100000
        index = -1
        for i in range(len(self.next_fields)):
            n_holes = self.next_fields[i].n_inaccessibles()
            if n_holes < minholes:
                minholes = n_holes
                index = i

        return self.moves[index]

    def print(self):
        import Tetris
        Tetris.Tetris.TILES[self.tile].print()
        print("Expected score : %f" % self.expectation)
        self.field.print()
        if len(self.moves) > 0:
            print("=>")
            self.next_fields[self.decision].print()
        else:
            print("GAME OVER")
        print()

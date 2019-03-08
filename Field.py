class Field:

    def __init__(self, n, m, grid=None, accessible=None, color=None):
        self.n = n
        self.m = m
        self.grid = grid
        self.accessible = accessible
        self.color = color

        if grid is None:
            self.grid = [[0 for x in range(self.m)] for y in range(self.n)]
            self.accessible = [self.n for i in range(0, self.m)]
            self.color = [['' for x in range(self.m)] for y in range(self.n)]

    def __copy__(self):
        new_grid = [[0 for x in range(self.m)] for y in range(self.n)]
        new_color = [[0 for x in range(self.m)] for y in range(self.n)]
        new_accessible = []

        for acc in self.accessible:
            new_accessible.append(acc)

        for i in range(0, self.n):
            for j in range(0, self.m):
                new_grid[i][j] = self.grid[i][j]
                new_color[i][j] = self.color[i][j]

        return Field(self.n, self.m, new_grid, new_accessible, new_color)

    # returns the exact representation of the grid
    def representation(self):
        hashes = ()
        for row in self.grid:
            hashes += (tuple(row).__hash__(),)
        return hashes.__hash__()
        # return tuple(self.accessible).__hash__()

    # returns the maximum height of a column and the average height
    def max_height(self):
        heights = [0 for j in range(0, self.m)]
        max_height = 0
        sum = 0

        for i in range(0, self.n):
            for j in range(0, self.m):
                if self.grid[i][j] == 1:
                    heights[j] = max(heights[j], self.n - i)

        for j in range(0, self.m):
            sum += heights[j]
            max_height = max(max_height, heights[j])

        return max_height, sum / self.m

    # returns the value-function approximation and its gradient
    def utility(self, w):
        max_height = 0
        min_height = self.n
        heights = [0 for j in range(0, self.m)]

        for i in range(0, self.n):
            for j in range(0, self.m):
                if self.grid[i][j] == 1:
                    heights[j] = max(heights[j], self.n-i)
                    max_height = max(max_height, heights[j])
                    min_height = min(min_height, heights[j])

        u = 0
        gradient = []
        # for j in range(0, self.m):
        #     u += w[j] * heights[j] #/ self.n
        #     gradient.append(heights[j]) # / self.n)

        avg_height = heights[0]
        for j in range(0, self.m-1):
            u += w[j] * (heights[j+1]-heights[j]) #/ self.n
            gradient.append((heights[j + 1] - heights[j]))# / self.n)
            avg_height += heights[j+1]
        avg_height /= self.m

        n_holes = self.n_inaccessibles()

        u += w[self.m - 1] * max_height #/ self.n
        u += w[self.m - 1] * min_height #/ self.n
        u += w[self.m + 1] * n_holes #/ (self.n * self.m)
        u += w[self.m + 2] * avg_height

        gradient.append(max_height) #/ self.n)
        gradient.append(min_height) #/ self.n)
        gradient.append(n_holes )#/ (self.n * self.m))
        gradient.append(avg_height)

        return u, gradient

    # updates the value-function approximation
    # cfr. n-step semi-gradient TD
    def utility_update(self, w, alpha, G):
        new_w = [0 for i in range(self.dimension())]
        est_utility, gradient = self.utility(w)
        absmax = 0

        for i in range(self.dimension()):
            new_w[i] = w[i] + alpha * (G - est_utility) * gradient[i]
            absmax = max(absmax, abs(new_w[i]))

        if absmax != 0:
            for i in range(self.dimension()):
                new_w[i] /= absmax

        return new_w

    # returns the dimension of the feature (and weight) vector
    def dimension(self):
        return self.m + 3

    # sets the cell (i, j) to the color
    def set(self, i, j, color):
        self.grid[i][j] = 1
        self.accessible[j] = min(self.accessible[j], i)
        self.color[i][j] = color

    # tells whether a tile can be placed a pos (i, j)
    def can_place(self, tile, i, j):
        for k in range(0, tile.n):
            for l in range(0, tile.m):
                if tile.get(k, l) == 1 and i + k >= self.accessible[j + l]:
                    return False
        return True

    # tells whether a tile can be shifted to left or right
    def can_shift(self, tile, i, j):
        for k in range(0, tile.n):
            for l in range(0, tile.m):
                if tile.get(k, l) == 1 and self.grid[i + k][j + l] == 1:
                    return False
        return True

    # returns all the positions where the tile and its rotations can be places
    def positions(self, cardinal_tile):
        pos = {}
        for rot in range(0, 4):
            tile = cardinal_tile.rotation(rot)
            for i in range(0, self.n-tile.n+1):
                for j in range(0, self.m-tile.m+1):
                    if self.can_place(tile, i, j):
                        pos[(j,rot)] = (i, j, rot)

            for z in range(0, self.m): # shift tiles when at bottom
                if (z, rot) in pos:
                    (i, oj, _) = pos[(z,rot)]
                    j = oj
                    while j > 0 and (j-1, rot) not in pos:
                        j -= 1
                        if self.can_shift(tile, i, j):
                            pos[(j, rot)] = (i, j, rot)
                        else:
                            break
                    j = oj
                    while j < self.m-1-tile.m and (j+1, rot) not in pos:
                        j += 1
                        if self.can_shift(tile, i, j):
                            pos[(j, rot)] = (i, j, rot)
                        else:
                            break
        return list(pos.values())

    # returns the grid after setting the tile w.r.t. the move
    # also returns the reward and the game gain
    def successor(self, tile, move):
        next = self.__copy__()
        gain, true_gain = next.set_tile(tile, move[0], move[1], move[2])
        return next, gain, true_gain

    # sets the tile at the given position and returns the reward and the game gain
    def set_tile(self, tile, i, j, rot):
        prev_holes = self.n_inaccessibles()
        prev_height, prev_avg = self.max_height()
        tile = tile.rotation(rot)
        for k in range(0, tile.n):
            for l in range(0, tile.m):
                if tile.get(k, l) == 1:
                    self.set(i+k, j+l, tile.color)

        row = self.n-1
        count = 0
        while row >= 0:
            row_full = True
            for j in range(0, self.m):
                if self.grid[row][j] == 0:
                    row_full = False
                    break
            if row_full:
                count += 1
                self.remove_row(row)
            else:
                row -= 1

        height, avg = self.max_height()
        holes = self.n_inaccessibles()
        value = 0
        if prev_holes - holes > 0:
            value += 1 * (prev_holes - holes)
        else:
            value += 1 * (prev_holes - holes)
        if prev_avg - avg > 0:
            value += 3 * (prev_avg - avg)
        else:
            value += 3 * (prev_avg - avg)

        import Tetris
        return value, 100 * ((count * Tetris.Tetris.ROW_GAIN) ** 2)

    # removes the row k
    def remove_row(self, k):
        for i in range(k, 0, -1):
            for j in range(0, self.m):
                self.grid[i][j] = self.grid[i-1][j]
                self.color[i][j] = self.color[i-1][j]
        for j in range(0, self.m):
            self.grid[0][j] = 0
            self.color[0][j] = 0
            self.accessible[j] = self.n
            start = self.accessible[j]
            for i in range(0, self.n):
                if self.grid[i][j] == 1:
                    self.accessible[j] = min(self.accessible[j], i)

    # computes the number of holes
    def n_holes(self):
        holes = 0
        self.vis = [[False for x in range(self.m)] for y in range(self.n)]
        for i in range(1, self.n):
            for j in range(0, self.m):
                if self.grid[i][j] == 0 and not self.vis[i][j]:
                    if not self.dfs(i, j):
                        holes += 1

        return holes

    # computes the number of inaccessible cells
    def n_inaccessibles(self):
        holes = 0
        for i in range(0, self.n):
            for j in range(0, self.m):
                if self.grid[i][j] == 0 and i >= self.accessible[j]:
                    holes += 1

        return holes

    # help function for n_holes
    def dfs(self, i, j):
        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            return False
        if self.vis[i][j]:
            return False
        if self.grid[i][j] == 1:
            return False

        self.vis[i][j] = True
        if self.dfs(i-1, j):
            return True
        if self.dfs(i+1, j):
            return True
        if self.dfs(i, j-1):
            return True
        if self.dfs(i, j+1):
            return True
        if i == 0:
            return True

        return False

    def print(self):
        for i in range(self.n):
            print('|', end='')
            for j in range(self.m):
                if self.grid[i][j] == 1:
                    if self.color[i][j] != "":
                        print(self.color[i][j] + "  " + '\x1b[0m', end='')
                    else:
                        print('\x1b[6;30;40m' + "  " + '\x1b[0m', end='')
                else:
                    print('  ', end='')
            print('|')
        print()

class Tile:

    def __init__(self, array, color, first=True):
        self.array = array
        self.n = len(array)
        self.m = len(array[0])
        self.rotations = []
        self.color = color

        if first:
            previous = self.rotate()
            for rot in range(0, 4):
                self.rotations.append(previous)
                previous = previous.rotate()

    # 0 if the cell is empty, 1 otherwise
    def get(self, i, j):
        return self.array[i][j]

    # returns a tile being this tile rotated by rot * 90°
    def rotation(self, rot):
        return self.rotations[rot]

    # returns a tile being this tile rotated by 90°
    def rotate(self):
        newArray = [[0 for j in range(self.n)] for i in range(self.m)]
        for i in range(self.n):
            for j in range(self.m):
                newArray[self.m-j-1][i] = self.array[i][j]
        return Tile(newArray, self.color, False)

    def print(self):
        for i in range(self.n):
            for j in range(self.m):
                val = "  "
                if self.array[i][j] == 1:
                    val = u"\u2588\u2588"
                print(val, end='')
            print()


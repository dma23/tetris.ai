class Block:
    def __init__(self, points, center, board):
        self._points = points
        self._center = center
        self._board = board

    
    def center_at(self, x, y):
        delta_x = x - self._center[0]
        delta_y = y - self._center[1]
        for p in range(len(self._points)):
            i = self._points[p][0] + delta_x
            k = self._points[p][1] + delta_y
            self._points[p] = (i, k)
        self._center = (x, y)


    def rotate_hard(self):
        new_points = list()
        for i in range(len(self._points)):
            x = self._points[i][0]
            y = self._points[i][1]
            new_x = self._center[0] + self._center[1] - y
            new_y = self._center[1] - self._center[0] + x
            new_points.append((new_x, new_y))
        self._points = new_points


    def rotate(self):
        new_points = list()
        for i in range(len(self._points)):
            x = self._points[i][0]
            y = self._points[i][1]
            new_x = self._center[0] + self._center[1] - y
            new_y = self._center[1] - self._center[0] + x
            new_points.append((new_x, new_y))

        if Block.in_bounds(new_points, self._board):
            self._points = new_points

    def down(self):
        new_points = list()
        for i in range(len(self._points)):
            x = self._points[i][0]
            y = self._points[i][1] + 1
            new_points.append((x, y))

        if Block.in_bounds(new_points, self._board):
            self._center = (self._center[0], self._center[1] + 1)
            self._points = new_points
            return True

        return False

    def left(self):
        new_points = list()
        for i in range(len(self._points)):
            x = self._points[i][0] - 1
            y = self._points[i][1]
            new_points.append((x, y))

        if Block.in_bounds(new_points, self._board):
            self._center = (self._center[0] - 1, self._center[1])
            self._points = new_points

    def right(self):
        new_points = list()
        for i in range(len(self._points)):
            x = self._points[i][0] + 1
            y = self._points[i][1]
            new_points.append((x, y))

        if Block.in_bounds(new_points, self._board):
            self._center = (self._center[0] + 1, self._center[1])
            self._points = new_points

    def get_points(self):
        return self._points


    @staticmethod
    def in_bounds(points, board):
        for p in points:
            if p[0] < 0 or p[0] > 9 or p[1] > 19 or p[1] < 0:
                return False
            if board[p[1]][p[0]] != 0:
                return False
        return True


class IBlock(Block):
    def __init__(self, board):
        super().__init__([(3,0), (4,0), (5,0), (6,0)], (4,0), board)
        self.block_type = 1
        self.rotations = 4


class OBlock(Block):
    def __init__(self, board):
        super().__init__([(4,0), (5,0), (4,1), (5,1)], (5,0), board)
        self.block_type = 2
        self.rotations = 1

    def rotate(self):
        return


class TBlock(Block):
    def __init__(self, board):
        super().__init__([(3,1), (4,1), (5,1), (4,0)], (4,1), board)
        self.block_type = 3
        self.rotations = 4


class SBlock(Block):
    def __init__(self, board):
        super().__init__([(3,1), (4,1), (4,0), (5,0)], (4,1), board)
        self.block_type = 4
        self.rotations = 2


class ZBlock(Block):
    def __init__(self, board):
        super().__init__([(3,0), (4,0), (4,1), (5,1)], (4,1), board)
        self.block_type = 4
        self.rotations = 2


class JBlock(Block):
    def __init__(self, board):
        super().__init__([(3,0), (3,1), (4,1), (5,1)], (4,1), board)
        self.block_type = 5
        self.rotations = 4


class LBlock(Block):
    def __init__(self, board):
        super().__init__([(5,0), (3,1), (4,1), (5,1)], (4,1), board)
        self.block_type = 5
        self.rotations = 4

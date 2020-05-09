class Score:

    score = 0
    level = 1
    lines_cleared = 0
    next_level = 5
    interval_speed = 0.5

    #def __init__(self, board):

    #   self.board = board

    def get_score(self):

        return self.score


    def get_level(self):

        return self.level


    def get_lines_cleared(self):
        return self.lines_cleared


    def get_interval(self):

        return self.interval_speed


    def reset_game(self):
        self.level = 1
        self.score = 0
        self.lines_cleared = 0
        self.next_level = 5
        return self.score, self.level, self.lines_cleared, self.next_level


    def update_score(self, rows):
        if rows != 0:
            self.score += Score.row_score(rows) * self.get_level()
            self.lines_cleared += rows
            self.update_level()
            #print(self.lines_cleared, self.next_level, self.level)
        #print(self.score)
        return self.score


    def update_level(self):

        if self.lines_cleared >= self.next_level and self.level < 16:
            self.level += 1
            self.next_level += 5
            self.interval_speed *= 0.9

    def scan_rows(self, board):
        # look for rows with no empty values
        row_count = 0
        for i in range(len(board)):
            row = board[i]
            if 0 not in row:
                row_count += 1

        # if there are any complete rows, update score
        if row_count > 0:
            self.lines_cleared += row_count
            return self.update_score(row_count)

    @staticmethod
    def row_score(row_count):

        if row_count == 1:
            return 10
        elif row_count == 2:
            return 25
        elif row_count == 3:
            return 50
        elif row_count == 4:
            return 100
        else:
            return 1




from random import choice
from copy import deepcopy
from math import sqrt

def to_tuple_of_tuples(lol):
    """Convert a list of lists into a tuple of tuples."""
    return tuple(tuple(row) for row in lol)

class SlidePuzzle(object):
    def __init__(self, n=3, grid_list=None):
        if grid_list is None:
            grid_list = range(1, n**2)
            grid_list.append(None)
        
        grid = [[None] * n for _ in range(n)]
        l = 0
        for i in range(n):
            for j in range(n):
                grid[i][j] = grid_list[l]
                if grid_list[l] is None:
                    self.space = (i, j)
                l += 1
        print(grid_list)
        print(grid)
                
        self.grid = grid
        # dicttionary with the goal positions
        tile_goals = {}
        goal_grid = [[None] * n for _ in range(n)]
        tile = 1
        for i in range(n):
            for j in range(n):
                if tile < n**2:
                    tile_goals[tile] = (i, j)
                    goal_grid[i][j] = tile
                tile += 1
        self.goal_grid = goal_grid
        self.tile_goals = tile_goals
        self.n = n

    def __str__(self):
        h_line = (2 * self.n + 1) * '-'
        s = [h_line]
        for i in range(self.n):
            line = ['|']
        for j in range(self.n):
            g = self.grid[i][j]
        if g:
            line.append(str(g))
        else:
            line.append(' ')
        line.append('|')
        s.append(''.join(line))
        s.append(h_line)

        return '\n'.join(s)

    def actions(self):
        acts = []
        i, j = self.space
        grid = self.grid
        n = self.n
        if i - 1 >= 0: # check to the left
            acts.append((i-1, j))
        if i + 1 < n: # check to the right
            acts.append((i+1, j))
        if j - 1 >= 0: # check up
            acts.append((i, j-1))
        if j + 1 < n: # check down
            acts.append((i, j+1))
            return acts

    def move(self, location):
        self.grid, self.space = self.next_state(self.grid, self.space, 
                                                location)
        
    def next_state(self, grid, space, location):
        i, j = location
        p, q = space
        grid[i][j], grid[p][q] = grid[p][q], grid[i][j]
        space = (i, j)

        return grid, space

    def shuffle(self, n_moves=1000):
        for _ in range(n_moves):
            self.move(choice(self.actions()))

    def is_solved(self):
        return self.grid == self.goal_grid

    def get_state(self, grid=None):
        if grid is None:
            grid = self.grid
            #return tuple(tuple(row) for row in grid)
        return to_tuple_of_tuples(grid)

    def h(self, grid=None):
        """Return the heuristic cost.

        Use the sum of the Manhattan distances from current position to the goal.

        If `grid` is None, use the current state grid. Otherwise use `grid`.
        """
        if grid is None:
            grid = self.grid
        h = 0
        for i in range(self.n):
            for j in range(self.n):
                tile = grid[i][j]
                if tile:
                    i_goal, j_goal = self.tile_goals[tile]
                    h += abs(i - i_goal) + abs(j - j_goal)
        return h
                

def solve(p):
    print('Solving')
    print(p)
    print()

    u = {}
    iters = 0
    while True:
        if p.is_solved():
            print('Solved in %d steps!' % iters)
            return locals()
        state = p.get_state()
        if state not in u:
            u[state] = p.h()
        actions = p.actions()
        print(actions)
        succs = [p.next_state(deepcopy(p.grid), p.space, a) for a in actions]
        m = float('inf')
        a_min = None
        for succ, a in zip(succs, actions):
            grid_tot = to_tuple_of_tuples(succ[0])
            if grid_tot not in u:
                u[grid_tot] = p.h(succ[0])
            if u[grid_tot] < m:
                m = u[grid_tot]
                a_min = a
        u[state] = 1 + m
        p.move(a_min)
        print(p)
        #raw_input()
        if iters == 100000:
            print('Too many iterations :(')
            return locals()
        iters += 1
                
def experiment_1():
    print('Testing slide puzzle')
    puzzle = SlidePuzzle()
    print(puzzle)
    puzzle.shuffle()
    n_moves = 5
    print(puzzle)
    for _ in range(n_moves):
        a =(puzzle.actions())
        puzzle.move(choice(a))
        print(puzzle)
    return locals()

def experiment_2():
    puzzle = SlidePuzzle(3)
    puzzle.shuffle(5)
    solve(puzzle)
    return locals()

def experiment_3():
    p = SlidePuzzle(grid_list=[7,8,6,5,3,2,1,None,4])
    solve(p)
    return locals()
    
if __name__ == '__main__':
    d = experiment_3()
    locals().update(d)

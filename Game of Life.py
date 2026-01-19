import numpy as np
import agentpy as ap

import matplotlib.pyplot as plt
import IPython


class Cell(ap.Agent):
    def setup(self):
        self.alive = self.model.random.random() < self.p.alive_prob
        self.next_alive = self.alive

    def initial_pos(self, space):
      self.space = space
      self.neighbors = space.neighbors 

    def update_state(self):
        nbs = self.neighbors(self, distance=1)
        live_nbs = 0
        for neighbor in nbs:
            if neighbor.alive:
                live_nbs +=1
        
        if self.alive:
            if live_nbs == 2 or live_nbs == 3:
                self.next_alive = True
            else:
                self.next_alive = False
        else:
            if live_nbs == 3:
                self.next_alive = True
            else:
                self.next_alive = False
        
    def new_state(self):
        self.alive = self.next_alive
    
class Game_of_Life(ap.Model):
    def setup(self):
        self.space = ap.Grid(self, (self.p.size, self.p.size), track_empty=False)

        self.cells = ap.AgentList(self, self.p.size * self.p.size, Cell)

        self.space.add_agents(self.cells, positions=[(i, j) for i in range(self.p.size) for j in range(self.p.size)])

        self.cells.initial_pos(self.space)

    def step(self):
        self.cells.update_state()
        self.cells.new_state()  

def animation_plot_single(m, ax):
    ax.clear()
    grid = np.zeros((m.p.size, m.p.size))

    for agent, pos in m.space.positions.items():
        x, y = pos
        grid[x, y] = 1 if agent.alive else 0
        
    ax.imshow(grid, cmap='binary_r', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    ax.set_xticks(np.arange(-0.5, m.p.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.p.size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(f"Conway's Game of Life - Step {m.t}", color='white')
    ax.set_facecolor('black')


def animation_plot(m, p):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    return IPython.display.HTML(animation.to_jshtml(fps=10))


parameters = {
    'size': 50,
    'steps': 60,
    'alive_prob': 0.2
}


animation_plot(Game_of_Life, parameters)

# references
# https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html
# https://pi.math.cornell.edu/~lipa/mec/lesson6.html
# https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
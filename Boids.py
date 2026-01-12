import numpy as np
import agentpy as ap

import matplotlib.pyplot as plt
import IPython


# the birds in this simulation are known as agents. each agent starts with random position
# and velocity which is stored as arrays

def normalise(v): # normalising the velocity vector
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm
    

class Boid(ap.Agent): # an agent which follows Reynold's three rules for swarm behaviour, with an
# additional fourth rule to remain within the simulation space
    def setup(self):
        
        self.velocity = normalise(self.model.nprandom.random(self.p.ndim) - 0.5)

    def initial_pos(self, space):
        
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos
        ndim = self.p.ndim

        # rule 1: cohesion

        nbs = self.neighbors(self, distance=self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_array = np.array(nbs.pos)
        nbs_vec_array = np.array(nbs.velocity)
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len # returns 2x2 array sums of each x component and y component of neighbours and averages
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # rule 2: separation

        v2 = np.zeros(ndim)

        for nb in self.neighbors(self, distance=self.p.inner_radius):
            v2 -= nb.pos - pos
        
        v2 *= self.p.separation_strength

        # rule 3: alignment

        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)
        
        # rule 4: simulation borders

        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength

        for i in range(ndim):
            if pos[i] < d:
                v4 += s
            elif pos[i] >  self.space.shape[i] - d: # checking for out of bounds
                v4[i]-=s
        
        # update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalise(self.velocity)

    def update_position(self):
        
        self.space.move_by(self, self.velocity)

class BoidsModel(ap.Model):
    def setup(self):
        
        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.initial_pos(self.space)

    def step(self):
        
        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction

def animation_plot_single(m, ax):
    ndim = m.p.ndim
    ax.set_title(f"Boids Flocking Model {ndim}D t={m.t}", c='white')
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # transform
    ax.scatter(*pos, s=1, c='white')
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    
    if ndim == 3:
        ax.set_zlim(0, m.p.size)
    ax.set_axis_off()

def animation_plot(m, p):
    projection = '3d' if p['ndim'] == 3 else None
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection=projection)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    return IPython.display.HTML(animation.to_jshtml(fps=20))


parameters = {
    'size': 50,
    'seed': 123,
    'steps': 200,
    'ndim': 3,
    'population': 200,
    'inner_radius': 3,
    'outer_radius': 10,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'separation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}


animation_plot(BoidsModel, parameters)


# references
# https://vergenet.net/~conrad/boids/pseudocode.html
# https://cs.stanford.edu/people/eroberts/courses/soco/projects/2008-09/modeling-natural-systems/boids.html
# https://www.cs.toronto.edu/~dt/siggraph97-course/cwr87/
# https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html


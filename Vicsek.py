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
    

class Vicsek(ap.Agent): 
    def setup(self):
        
        self.velocity = normalise(self.model.nprandom.random(self.p.ndim) - 0.5) # normalised velocity makes the speed constant 1
        self.theta = np.arctan2(self.velocity[1], self.velocity[0])
        self.theta_next = self.theta

    def initial_pos(self, space):
        
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos
        ndim = self.p.ndim

        # rules of motion
        nbs = self.neighbors(self, distance=self.p.interaction_radius)
        nbs_len = len(nbs)
        noise = (np.random.random() - 0.5) * self.p.noise_strength
        sum_cos = np.cos(self.theta)
        sum_sin = np.sin(self.theta)

        for nb in nbs:
            nb_velocity = nb.velocity
            nb_theta = np.arctan2(nb.velocity[1], nb.velocity[0])
            nb_cos = np.cos(nb_theta)
            nb_sin = np.sin(nb_theta)
            sum_cos += nb_cos
            sum_sin += nb_sin
        
        total_nbs = nbs_len + 1
        avg_cos = sum_cos / total_nbs
        avg_sin = sum_sin / total_nbs
        avg_theta = np.arctan2(avg_sin, avg_cos)
        new_theta = (avg_theta + noise) % (2*np.pi)

        self.theta = new_theta
        self.theta_next = new_theta
        self.velocity = np.array([np.cos(new_theta),np.sin(new_theta)])
        
    def update_position(self):
        
        old_pos = self.pos.copy()
        new_pos = old_pos + self.velocity
        
        
        for i in range(self.p.ndim): # periodic boundary conditions
            while new_pos[i] >= self.p.size:
                new_pos[i] -= self.p.size
            while new_pos[i] < 0:
                new_pos[i] += self.p.size

        self.pos = new_pos
        self.space.move_to(self, new_pos)

class VicsekModel(ap.Model):
    def setup(self):
        
        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Vicsek)
        self.space.add_agents(self.agents, random=True)
        self.agents.initial_pos(self.space)

    def step(self):
        
        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction

def animation_plot_single(m, ax):
    ndim = m.p.ndim
    ax.set_title(f"Vicsek Flocking Model {ndim}D t={m.t}", c='white')
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # transform
    ax.scatter(*pos, s=1, c='white')
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    
def animation_plot(m, p):
    projection = '3d' if p['ndim'] == 3 else None
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection=projection)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    model_instance = m(p)
    animation = ap.animate(model_instance, fig, ax, animation_plot_single)
    return IPython.display.HTML(animation.to_jshtml(fps=20))


parameters = {
    'size': 50,
    'seed': 123,
    'steps': 200,
    'ndim': 2,
    'population': 400,
    'interaction_radius': 1,
    'noise_strength': 0.3
}


animation_plot(VicsekModel, parameters)


# references
# https://web.mit.edu/8.334/www/grades/projects/projects10/Hernandez-Lopez-Rogelio/dynamics_2.html
# https://grokipedia.com/page/Vicsek_model
# https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html


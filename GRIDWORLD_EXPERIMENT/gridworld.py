"""
The first experiment will consist of attempting to let an agent navigate a so-called gridworld. 
For a grid of size (n,n), the goal is for the agent to learn a policy that enables it to go from the starting point to a target. 
The target will be in a fixed location, and the agent will start at a random location in this grid. 

To make this environment, the gymnasium library is used, a tutorial can be found here: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
This file largely follows that tutorial for now
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt
from utils import state_transition_kernel, MC_transition_matrix

class Gridworld(gym.Env):
    
    # Two ways to show the environment, a text output or a matplotlib plot
    metadata = {"render_modes": ["text", "plot"], "render_fps": 4}

    def __init__(self, render_mode = None, size = 5, continuing = False, stochastic = False):
        """
        Initializing the environment

        Params: 
            render_mode: how should the environment be displayed?
            size: the size of the gridworld, there are size^2 grids
            continuing: do we have a continuing environment (infinite-horizon?)
            stochastic: the environment can be made stochastic by including one or two grids that are like ice, upon reaching them, 
            the next action is random. The independent coupling would still work. 
        """

        self.size = size

        # Do we consider a continuing environment?
        self.continuing = continuing

        # States are tuples (grids) within the gridworld
        self.observation_space = gym.spaces.Box(low = 0,high=size-1, shape=(2,), dtype=int)

        # The agent has four options: moving up, down, left or right
        self.action_space = gym.spaces.Discrete(5)

        # Now we define what chosen action corresponds to what direction
        self._action_to_direction = {
            0: np.array([0,0]), # stay
            1: np.array([1, 0]), # right
            2: np.array([0, 1]), # up
            3: np.array([-1, 0]), # left
            4: np.array([0, -1]) # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        ##################
        # Define some variables that should be easily accessible from the environment
        # the action and state space for example, as well as the state transition kernel p(s'|s,a)
        # Another object that should be easily accessible is the reward for a given state -> separate function
        ##################

        states = list(range(size))
        states = [(i, j) for i in states for j in states]
        self.S = states

        actions = list(range(5))
        self.A = actions

        self.state_transition_kernel = state_transition_kernel(size=size)

        # For now, the target location is fixed in the top right corner of the grid
        self._target_location = np.array([self.size-1,self.size-1])

        self.stochastic = stochastic

        
    def _get_reward(self, state):
        # computing the distance to the target, the reward is 1 if the agent reached the target, 0 otherwise
        distance = np.linalg.norm(state - self._target_location, ord=1)
        return 1 if distance==0 else 0


    # A simple method that allows us to quickly get the location of our agent
    def _get_obs(self):
        return self._agent_location
    

    def reset(self, initial_location, seed = None):
        """
        Called at the start of each episode. This corresponds to running a new markov chain from the start. 

        Params: 
            initial_location: a numpy array or list of length 2 that gives the initial location
        """
        # This initializes the environments random number generator
        super().reset(seed=seed)

        # make sure that the agent is located inside the grid
        assert 0 <= initial_location[0] < self.size and 0 <= initial_location[1] < self.size

        self._agent_location = np.array(initial_location)
        obs = self._get_obs()
        return obs
    
    def step(self,action):
        # Each action corresponds to a direction
        direction = self._action_to_direction[action]

        # We do not want to leave the grid, so when at the right of the grid, moving right will not do anything. 
        # In the environment creation we can do a simple np.clip operation, but this 
        # is something we will have to take care of in the transition matrix of our Markov Chain
        
        self._agent_location = np.clip(
        self._agent_location + direction, 0, self.size - 1
        )

        # If the environment is stochastic and the agent wants to go to one of two icy grids, 
        # a random action is chosen. 
        if (self.stochastic and (np.array_equal(self._agent_location, np.array([2,2])) or np.array_equal(self._agent_location, np.array([4,4])))):
            action = np.random.randint(1,5) # Choose a random action (not staying put)
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
                )
            
        
        # Check if the agent reached the goal state
        goal_reached = np.array_equal(self._agent_location, self._target_location)

        # We can terminate when the agent reaches the target or we have a continuing environment
        if not self.continuing:
            terminated= goal_reached
        else:
            # We will not terminate the episode upon reaching the goal state
            terminated = False 

        # Apart from reaching the target, the episode could also be ended after a fixed number of timesteps
        # for now we do not do that
        truncated = False

        # We also have to come up with a reward structure
        # The agent receives a reward of 1 when reaching the target, 0 otherwise
        reward = 1 if goal_reached else 0

        if self.continuing and goal_reached:
            # We currently do not spend a single moment in this state, and just get the reward
            self._agent_location = np.array([0,0])

        observation = self._get_obs()

        # In general environments, the info object could contain extra information about the environment such as the 
        # number of lives left for the agent in an Atari game, not needed here
        info = None

        return observation, reward, terminated, truncated, info


    def render_text(self, d=1):
        grid = np.zeros((self.size, self.size), dtype=int)
        grid = grid.astype(str)
        grid[grid == '0'] = '.'
        grid[(self.size-1)-self._agent_location[1], self._agent_location[0]] = 'S'
        grid[(self.size-1)-self._target_location[1], self._target_location[0]] = 'T'
        print(f"{grid} \n")


    def render_plot(self):
        # Create a figure with a white background
        fig = plt.figure(figsize=(self.size, self.size), facecolor='white')

        # Set up the axis and grid
        ax = plt.gca()
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        plt.xticks(np.arange(0.5, self.size, step=1))
        plt.yticks(np.arange(0.5, self.size, step=1))
        ax.set_aspect('equal', 'box')

        # Create a matrix representing the grid
        grid_matrix = np.zeros((self.size, self.size))  # Initialize with white (0)

        # Set specific grid cells to a different value for coloring (light blue)
        grid_matrix[2, 2] = 0.2  # (2,2) 
        grid_matrix[4, 4] = 0.2  # (4,4) 

        # Display the grid with specified colors using imshow() with a custom colormap
        plt.imshow(grid_matrix, cmap='Blues', origin='lower', extent=[-0.5, self.size-0.5, -0.5, self.size-0.5])

        plt.grid()
        plt.plot(self._agent_location[0], self._agent_location[1], 'ro', markersize=15)
        plt.plot(self._target_location[0], self._target_location[1], 'bs', markersize=15)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.show()


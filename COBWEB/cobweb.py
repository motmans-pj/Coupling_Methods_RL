import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Cobweb(gym.Env):
    """
    The environment in the Cobweb model experiment.
    """

    def __init__(self, p_competitive, p_monopoly, demand_intercept, demand_slope, supply_intercept, supply_slope):
        """
        Defining what the Cobweb market looks like. 

        Params: 
            p_competitive: the price equal to marginal cost, prices can not be lower. 
            p_monopoly: the price a monopoly firm would charge, prices can not be higher 
                (similar pricing space as used by Calvano et al. 2020). 

            For the demand D_t = a_0 - a_1 p_t
                The demand_intercept = a_0
                The demand_slope is a_1

            For the supply S_t = b_0 + b_1 p_t
                The supply intercept is b_0
                The supply slope is b_1
        """

        self.p_min = p_competitive
        self.p_max = p_monopoly
        # Defining the state and action space
        #self.S = np.linspace(self.p_min, self.p_max,num = 16)
        # For this specific experiment, easy to work with multiples of 3
        self.S = np.linspace(0,15, num=16)/3
        self.A = self.S
        self.a0 = demand_intercept
        self.a1 = demand_slope
        self.b0 = supply_intercept
        self.b1 = supply_slope
        # The environment will at all times be able to return the current observation
        self.p_t = None

    def reset(self, p0):
        """
        Resets the price to p0
        """
        self.p_t = p0
        return self.p_t

    def step(self, a_t):
        """
        Set a step in the environment. at is the action at time t, the expectation made at time t for the next price
        """
        # Think more about the shock
        shock = np.random.binomial(5,0.5) * np.random.choice([1,-1]) * (self.S[1]-self.S[0])
        # The environment transition that arises from the three structural equations
        self.p_t = (self.a0-self.b0)/self.a1 - self.b1/self.a1 * a_t + shock
        # set p to the closest value that is allowed in the state space
        self.p_t = min(self.S, key=lambda num: abs(num - self.p_t))
        # The reward
        self.r_t = - (self.p_t - a_t)**2
        return self.p_t, self.r_t
    


        

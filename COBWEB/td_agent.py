''''
Implementation from the paper Learning and Planning in Average-Reward MDP's by Wan, Naik and Sutton
'''

import numpy as np
from cobweb import Cobweb
from matplotlib import pyplot as plt
import pandas as pd
import math
import toleranceinterval as ti
from tqdm import tqdm 

class TD_Agent:

    def __init__(self,initial_step_size, method=None,mu=None,rho=None,phi_0=None, phi_1=None, p_competitive=0, p_monopoly=5, demand_intercept=10, demand_slope=2,supply_intercept=5, supply_slope=1):
        # Parameters for the AR1 process of the policy
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        # Parameters if policy weights REE and naive expectations
        self.mu = mu
        self.rho = rho

        self.method = method
        # The step size is a hyperparameter
        self.initial_step_size = initial_step_size

        # Initializing the economy
        self.cobweb = Cobweb(p_competitive,p_monopoly,demand_intercept, demand_slope, supply_intercept, supply_slope)
        self.eta = 1 # Default value in their implementation

    def update_step_size(self, t):
        # Step size for simple average as 1/t
        self.step_size = 1/t

    def policy(self, s_t):
        """
        The AR(1)-process. The agent makes his expectation, and then predicts outside shocks. 
        """
        shock = np.random.binomial(2, 0.2) * np.random.choice([1,-1]) * (self.cobweb.S[1]-self.cobweb.S[0])
        if self.method == "AR1":
            a_t = self.phi_0 + self.phi_1 * s_t + shock
            a_t = min(self.cobweb.S, key=lambda num: abs(num - a_t))
        elif self.method == "Weighted":
            a_t = self.rho * self.mu + (1-self.rho) * s_t
            a_t = min(self.cobweb.S, key=lambda num: abs(num - a_t))
        return a_t


    def TD_learning(self, num_steps, log=False):
        """
        This function performs one 'episode' of the TD learning algorithm
        """
        # Initialize the state randomly
        s_0 = np.random.choice(self.cobweb.S)
        self.cobweb.reset(s_0)
        # Initialize the estimates
        V = {state:0 for state in self.cobweb.S}
        gain_estimate = 0
        gain_estimates = []
        # Initialize the step size
        self.step_size = self.initial_step_size
        s_t = s_0
        t=0
        # Trajectory is: 
        # (s_0, a_0, s_1, r_1, a_1,s_2, r_2, a_2,...)
        for t in range(1,num_steps):
            # Deciding on the action
            a_t = self.policy(s_t)
            # Update the timestep after taking the action
            t +=1
            # Take a step in the environment
            next_s, r = self.cobweb.step(a_t)
            # Performing the update
            TD_error = r - gain_estimate + V[next_s] - V[s_t]
            if log:
                print(f"Value s_t: {self.V[s_t]}")
                print(f"Value next state: {self.V[next_s]}")
                print(f"Expected price: {a_t}")
                print(f"The true price: {next_s}")
                print(f"Reward: {r}")
                print(f"TD error: {TD_error}")
            V[s_t] += self.step_size * TD_error
            gain_estimate += self.eta * self.step_size * TD_error
            gain_estimates.append(gain_estimate)
            self.update_step_size(t)
            s_t = next_s
        return V, gain_estimate, gain_estimates
    
    def TD_tol_interval_gain(self, num_runs, max_steps, plot=True):
        all_estimates = []
        for run in range(num_runs):
            V, gain, gain_estimates = self.TD_learning(num_steps = max_steps)
            all_estimates.append(gain_estimates)
        all_estimates = np.array(all_estimates)
        means = np.mean(all_estimates, axis=0)
        std = np.std(all_estimates, axis = 0)
        # Assuming normal distribution as underlying distribution
        bounds = ti.twoside.normal(all_estimates.transpose(), 0.9, 0.95)
        lower_bounds_ti = bounds[:,0]
        upper_bounds_ti = bounds[:,1]

        final_estimate = means[-1]
        standard_deviation =std[-1]
        if plot:
            plt.plot(np.arange(len(means)), means, color='black', label=f'Final estimate of {round(final_estimate,4)} with std = {round(standard_deviation, 4)}')
            plt.fill_between(np.arange(len(means)), lower_bounds_ti, upper_bounds_ti, color='gray', alpha=0.5, label='Tolerance Interval')
            plt.xlabel('Restarts')
            plt.ylabel('Gain Estimate')
            plt.title(f'Tolerance interval for gain estimation over {num_runs} runs.')
            plt.legend()
            # Adjust ylim on plot, shows drastic tolerance interval after only very few restarts
            plt.ylim([lower_bounds_ti[500], upper_bounds_ti[500]])
            plt.grid(True)
            plt.show()
        return final_estimate, standard_deviation
    
    def TD_Histogram_state_value(self, num_runs, max_steps, s):
        estimates = []
        for run in range(num_runs):
            V, gain, gain_estimates = self.TD_learning(num_steps=max_steps)
            estimates.append(V[s])

        # Calculate mean and standard deviation for better range selection
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)

        # Define the range centered around the mean with a certain standard deviation range
        range_min = mean_estimate - 2 * std_estimate
        range_max = mean_estimate + 2 * std_estimate

        # Filter values within the selected range
        filtered_estimates = [est for est in estimates if range_min <= est <= range_max]

        fig, ax = plt.subplots()
        ax.hist(filtered_estimates, bins=30)  # Adjust the number of bins as needed
        ax.set_title(f"Estimates for the value of price {round(s, 3)} over {num_runs} runs")
        ax.set_xlabel('Value estimate')
        ax.set_xlim(range_min, range_max)  # Set x-axis limits to the selected range
        plt.show()

        print(f"The mean estimate is {mean_estimate}")
        print(f"The standard deviation of the estimate is {std_estimate}")






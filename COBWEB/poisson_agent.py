import numpy as np
from cobweb import Cobweb
from matplotlib import pyplot as plt
import pandas as pd
import math
import toleranceinterval as ti
from tqdm import tqdm

class PoissonAgent:

    def __init__(self, method = None, mu=None, rho=None, phi_0 = None, phi_1 = None, p_competitive=0, p_monopoly=5, demand_intercept=10, demand_slope=2,supply_intercept=5, supply_slope=1):

        """
        Params: 

            The agent uses an AR1 process to predict next year's price
            p^e_t = phi_0 + phi_1 p_{t-1}
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
        # If we do an AR(1) for p^e_t = phi_0 + phi_1 p_{t-1}
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        # If p^e_t = \rho * \mu + (1-\rho) * p_{t-1}
        self.rho = rho
        self.mu = mu
        self.method = method

        self.cobweb_X = Cobweb(p_competitive=p_competitive,p_monopoly=p_monopoly,demand_intercept=demand_intercept,demand_slope=demand_slope,supply_intercept=supply_intercept, supply_slope=supply_slope)
        self.cobweb_Y = Cobweb(p_competitive=p_competitive,p_monopoly=p_monopoly,demand_intercept=demand_intercept,demand_slope=demand_slope,supply_intercept=supply_intercept, supply_slope=supply_slope)

    def policy(self, s_t):
        """
        The policy, either an AR(1) or weight of rational and naive expectations
        """
        shock = np.random.binomial(2, 0.2) * np.random.choice([1,-1]) * (self.cobweb_X.S[1]-self.cobweb_X.S[0])
        if self.method == "AR1":
            a_t = self.phi_0 + self.phi_1 * s_t + shock
            a_t = min(self.cobweb_X.S, key=lambda num: abs(num - a_t))
        elif self.method == "Weighted":
            a_t = self.rho * self.mu + (1-self.rho) * s_t
            a_t = min(self.cobweb_X.S, key=lambda num: abs(num - a_t))
        return a_t
    
    def gain_tol_interval(self, max_restarts ,num_runs,L, k,l, plot=True):
        """
        Plot tolerance intervals for the estimation of the gain

        Params: 
            max_restarts: the number of simulated couplings in a single run
            num_runs: the number of independent runs to compute tolerance intervals
        """
        all_estimates = []
        total_cost = 0
        for run in range(num_runs):
            estimates = []
            running_estimate=0
            for num_restarts in range(1,max_restarts):
                Hkl, _, cost = self.sample_unbiased_estimator(L=L,k=k,l=l)
                total_cost += cost
                running_estimate = running_estimate + (1/num_restarts) * (Hkl - running_estimate)
                estimates.append(running_estimate)
            all_estimates.append(estimates)
        all_estimates = np.array(all_estimates)
        means = np.mean(all_estimates, axis=0)
        standard_deviations = np.std(all_estimates, axis=0)
        # Assuming normal distribution as underlying distribution
        bounds = ti.twoside.normal(all_estimates.transpose(), 0.9, 0.95)
        lower_bounds_ti = bounds[:,0]
        upper_bounds_ti = bounds[:,1]
        final_estimate = means[-1]
        standard_deviation = standard_deviations[-1]
        if plot:
            plt.plot(np.arange(len(means)), means, color='black', label=f'Final estimate of {round(final_estimate,4)} with std = {round(standard_deviation,4)}')
            plt.fill_between(np.arange(len(means)), lower_bounds_ti, upper_bounds_ti, color='gray', alpha=0.5, label='Tolerance Interval')
            plt.xlabel('Restarts')
            plt.ylabel('Gain Estimate')
            plt.title(f'Tolerance interval for gain estimation over {num_runs} runs.')
            plt.legend()
            # Adjust ylim on plot, shows drastic tolerance interval after only very few restarts
            plt.ylim([lower_bounds_ti[20], upper_bounds_ti[20]])
            plt.grid(True)
            plt.show()
        return final_estimate, standard_deviation, total_cost


    def sample_unbiased_estimator(self,k,l,L, X_0 = "random", Y_0 = "random"):
        """
        Calculate the unbiased estimator for the stationary expectation = the gain of the policy. 
        The estimator is calculated "on the fly", while simulating the chains. 

        States are X_t and Y_t, actions are X_a_t and Y_a_t for X and Y respectively

        Adapted from https://github.com/pierrejacob/unbiasedpoisson/blob/master/R/sample_unbiasedestimator.R
        """

        # Initializing both chains
        if X_0 == "random":
            X_0 = np.random.choice(self.cobweb_X.S)
            # The first state has no associated reward
            X_r_t = 0
        if Y_0 == "random":
            Y_0 = np.random.choice(self.cobweb_Y.S)
            Y_r_t = 0
        X_t = self.cobweb_X.reset(X_0)
        Y_t = self.cobweb_Y.reset(Y_0)

        time = 0
        cost = 0
        mcmcestimator = 0
        bias_correction = 0

        for t in range(1,L+1):
            # The action at time t-1 depends on the state at time t-1
            X_a_t = self.policy(X_t)
            time += 1
            X_t, X_r_t = self.cobweb_X.step(X_a_t)
            cost += 1
            if time>=k:
                mcmcestimator += X_r_t
        if time >= (k+L):
                bias_correction += (math.floor((time-k)/L)-math.ceil(max(L,time-l)/L)+1) * (X_r_t-Y_r_t)

        # At this point time=L, giving us X_L and Y_0, now we generate (X_t) and (Y_{t-L}) until 
        # X_t = Y_{t-L}
        meeting_time = math.inf
        while time < (max(meeting_time, l)):
            time += 1
            # If the meeting time is infinite, this means the chains did not yet meet
            if math.isinf(meeting_time):
                # If they did not meet yet, run them according to the coupled kernel
                # ie independent until meeting
                X_a_t, Y_a_t = self.policy(X_t), self.policy(Y_t)
                X_t, X_r_t = self.cobweb_X.step(X_a_t)
                Y_t, Y_r_t = self.cobweb_Y.step(Y_a_t)
                cost += 2
                if X_t == Y_t:
                    meeting_time = time

                if (k<=time<=l):
                    mcmcestimator += X_r_t
                
                if (time >= k+L):
                    bias_correction += (math.floor((time-k)/L)-math.ceil(max(L,time-l)/L)+1) * (X_r_t-Y_r_t)
            # If the chains already met, but chain_X does not yet have l components
            # we run a single chain forward, because the state will be the same afterwards
            else:
                X_a_t = self.policy(X_t)
                X_t, X_r_t = self.cobweb_X.step(X_a_t)
                cost += 1
                if (k<=time<=l):
                    mcmcestimator += X_r_t
                
        unbiased_estimator = (mcmcestimator + bias_correction)/(l-k+1)

        return unbiased_estimator, meeting_time, cost
    
    def sample_value_estimate(self, s, sref):
        "Initialize both chains to their initial prices"
        X_p_t = self.cobweb_X.reset(s)
        Y_p_t = self.cobweb_Y.reset(sref)
        Vxy = 0
        met=False
        while not met:
            a_X_t, a_Y_t = self.policy(X_p_t), self.policy(Y_p_t)
            X_p_t, X_r_t = self.cobweb_X.step(a_X_t)
            Y_p_t, Y_r_t = self.cobweb_Y.step(a_Y_t)
            Vxy += X_r_t - Y_r_t
            if X_p_t == Y_p_t:
                met=True
        return Vxy


    def construct_tol_interval_value(self, num_runs, num_restarts, s, sref):
        all_estimates = []
        for run in tqdm(range(num_runs)):
            estimates = []
            running_estimate=0
            for restart in range(1,num_restarts):
                Vxy = self.sample_value_estimate(s,sref)
                running_estimate = running_estimate + (1/restart) * (Vxy - running_estimate)
                estimates.append(running_estimate)
            all_estimates.append(estimates)
        all_estimates = np.array(all_estimates)
        means = np.mean(all_estimates, axis=0)
        std = np.std(all_estimates, axis=0)
        # Assuming normal distribution as underlying distribution
        bounds = ti.twoside.normal(all_estimates.transpose(), 0.9, 0.95)
        lower_bounds_ti = bounds[:,0]
        upper_bounds_ti = bounds[:,1]

        plt.plot(np.arange(len(means)), means, color='black', label=f'Final estimate of {round(means[-1],4)}')
        plt.fill_between(np.arange(len(means)), lower_bounds_ti, upper_bounds_ti, color='gray', alpha=0.5, label='Tolerance Interval')
        plt.xlabel('Restarts')
        plt.ylabel('Gain Estimate')
        plt.title(f'Tolerance interval for value estimate of state {round(s,3)}.')
        plt.ylim([lower_bounds_ti[20], upper_bounds_ti[20]])
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_chains(self, X_p_0, Y_p_0,L,l, plot_chains=False):
        """
        Simulates two coupled lagged chains.

        Simulates two chains, X and Y, with an independent coupling until they meet. The chains evolve based on the
        given initial states and lag parameter.

        Parameters:
        - X_p_0: Initial price for chain X. If "random", a random price
        is uniformly selected. Otherwise, provide the initial price
        - Y_p_0 (str or object): Initial state for chain Y. If "random", a random state
        is selected. Otherwise, provide the initial price.
        - L (int): The lag parameter.
        - l (int): The highest index for which r(X_l) is included 

        Returns:
        - chain_X (list): List containing states of chain X at each time step, including the
        initial state.
        - chain_Y (list): List containing states of chain Y at each time step, including the
        initial state.
        - tau (int): meeting time of the chains (number of elements in Y)
        """
        if X_p_0 == "random":
            X_p_0 = np.random.choice(self.cobweb_X.S)
        if Y_p_0 == "random":
            Y_p_0 = np.random.choice(self.cobweb_Y.S)
        # Keep track of the meeting time
        tau = 0
        # Reset both economies to their initial states
        X_p_t = self.cobweb_X.reset(X_p_0)
        Y_p_t = self.cobweb_Y.reset(Y_p_0)

        if plot_chains:
            states_X = [X_p_t]
            states_Y = [Y_p_t]

        rewards_X = []
        rewards_Y = []

        for t in range(1,L+1):
            a_t = self.policy(X_p_t)
            X_p_t, r_t = self.cobweb_X.step(a_t)
            if plot_chains:
                states_X.append(X_p_t)
            rewards_X.append(r_t)

        # If they start out equal, we still want to simulate because 
        # the meeting time is defined for t>1
        equal_start = Y_p_t == X_p_t
        while (X_p_t!= Y_p_t) or equal_start:
            # If they started equal, we do not want that to lead to an infinite while loop
            equal_start = False

            X_a_t, Y_a_t = self.policy(X_p_t), self.policy(Y_p_t)
            X_p_t, X_r_t = self.cobweb_X.step(X_a_t)
            Y_p_t, Y_r_t = self.cobweb_Y.step(Y_a_t)

            if plot_chains:
                states_X.append(X_p_t)
                states_Y.append(Y_p_t)
            
            rewards_X.append(X_r_t)
            rewards_Y.append(Y_r_t)

        # Then run at least until l

        while len(rewards_X) < l:
            a_t = self.policy(X_p_t)
            X_p_t, r_t = self.cobweb_X.step(a_t)
            if plot_chains:
                states_X.append(X_p_t)
            rewards_X.append(r_t)

        if plot_chains:

            fig, ax = plt.subplots()
            x = list(range(len(rewards_X)+1))
            ax.plot(x,states_X, label = 'Market X')
            ax.plot(x,[np.nan] * L + states_Y + [np.nan]*(len(x)- (L+len(states_Y))), label = 'Market Y')
            ax.legend()
            ax.set_title('Price evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Price')
    

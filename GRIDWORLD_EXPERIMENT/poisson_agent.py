from utils import *
from gridworld import Gridworld
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import toleranceinterval as ti

class PoissonRL:
    """
    The coupling method to solving the Poisson Equation

    Params: 
        env_X (a gridworld environment): the environment in which our agent X will operate (1 chain)
        env_Y (a gridworld environment): the environment in which our agent Y will operate (1 chain)
        initial_policy: the initial policy, represented as a dataframe of n_states rows and 5 (n_actions) columns
                it is this policy that we will evaluate and improve upon iteratively
        K: the transition matrix for the Markov Chain that arises because of some policy pi, it is a square dataframe
        Gy: an estimate of the bias function


    Methods: 

        gamma_coupling: a maximal coupling for the kernels K(x->.) and K(y->.)
        sample_single_state: samples the next state for a single chain according to kernel K
        sample_coupled_states: samples the next state using a maximal coupling of the transition kernel K 
        simulate_coupled_lagged_chains: simulate two chains with a lag that are coupled

        gain_estimation: estimate the gain of a given policy (average reward when in steady-state)
        evaluate_policy: evaluates the current policy, 'a prediction task'

    """

    def __init__(self, env_X, env_Y, initial_policy) -> None:

        """
        Params: 
            env_X (a gridworld environment): the environment in which our agent X will operate (1 chain)
            env_Y (a gridworld environment): the environment in which our agent Y will operate (1 chain)
            initial_policy: the initial policy, represented as a dataframe of n_states rows and 5 n_actions columns
                it is this policy that we will evaluate
        """
        # X and Y should operate in the same grid
        assert env_X.S == env_Y.S
        self.env_X = env_X
        # X and Y will also have the same state_transition_kernel
        self.state_transition_kernel = env_X.state_transition_kernel
        self.env_Y = env_Y
        
        # X and Y will follow the same policy
        self.policy = initial_policy

        # X and Y will also have the same stochastic transition matrix K
        self.K = MC_transition_matrix(self.state_transition_kernel, self.policy)



    def gain_estimation(self,L,k,l,method="main", coupling="maximal"):
        """
        Implementation of gain estimation: H_k:l(X,Y), the chains are simulated (and stored) in this function
        So two chains are stored in memory 

        Params: 
            the lag L
            k: the lowest index for which we include X_k
            l: the highest index for which we include X_l
            method (str): 'main' or 'appendix', specifies how the estimator for \pi(h) should be calculated
            coupling: whether we use the maximal coupling, or an independence coupling
        """
        # define k: when do we start? 
        # initial X and Y randombly distributed

        chain_X, chain_Y, tau = self.simulate_coupled_lagged_chains(initial_X="random", initial_Y="random", L = L, coupling = coupling)
        # An efficient implementation to calculate h(x_t) for each element of the chain
        rewards_X, rewards_Y = list(map(self.env_X._get_reward, chain_X)), list(map(self.env_Y._get_reward, chain_Y))


        if method == "main":
            
            HL_kl = 0
            for i in range(k, l+1):
                if i-1 < len(rewards_X):
                    # if i = 3, we want to select the third element, which we do with index i-1=2 in python
                    HL_k = rewards_X[i-1] + sum([rewards_X[i-1 + j*L] - rewards_Y[i-1 + (j-1)*L] for j in range(1,len(chain_X)) if i-1 + j*L < len(chain_X) and i-1 + (j-1)*L < len(chain_Y)])
                    HL_kl += HL_k
            
            return (1/(l-k+1)) * HL_kl, tau


        elif method == "appendix":
            HL_kl = 0
            for i in range(k,l+1):
                if i-1 < len(rewards_X):
                    BC_i_to_l = 0
                    # meeting time: calculated from chain X or Y?
                    for t in range(k+L, len(chain_X)):
                        vt = math.floor((t-k)/L) - math.ceil(max(L,t-l)/L) + 1
                        BC_i_to_l += vt/(l-k+1) * (rewards_X[t-1] - rewards_Y[t-1-L])

                    HL_kl += rewards_X[i-1] + BC_i_to_l

            return (1/(l-k+1)) * HL_kl, tau     
        

        
            

    def value_estimation(self, plot_convergence = False, max_restarts = 200, epsilon = np.inf, initial_state_Y = (0,0), coupling = "maximal", tol_interval_computation = False, evaluation_state=None):
        """
        Running the algorithm to estimate the value for each state

        Params: 
            plot_convergence: whether we want to show plots of the estimated bias value for each initial state
            max_restarts: the maximum number of times we will restart the couplings from each starting state for agent X
            epsilon: a stopping rule, if the change in bias estimate is lower than epsilon, we stop the run for the state. 
            initial_state_Y: the initial state from which agent Y will start, either a fixed state or a distribution (randomized)
            coupling: whether to do a maximal coupling (for use in deterministic environment only) 
                      or an independent coupling (also in stochastic environment)
            tol_interval_computation: whether we want to compute the tolerance interval of a specific state -> we want to return the estimates 
                over runs for a single state
            evaluation_state: the state for which we want to compute the tolerance interval. 

        """

        # when evaluating a new policy, we reset the value function
        Gy = pd.DataFrame(np.zeros(len(self.env_X.S)), index = self.env_X.S, columns = ['Bias'])

        if tol_interval_computation:
            # if we want to evaluate one state
            states = [evaluation_state]
        else:
            # if we want to estimate the whole value function
            states = self.env_X.S

        # We will make X start from every grid to get a full estimate of Gyx for each x
        for initial_state_X in states:
            # Estimates will keep the estimate for the bias of the initial state at each iteration
            estimates = []
            if plot_convergence:
                # If we want to plot the convergence
                fig, axs = plt.subplots(nrows = 1, figsize = (3,3))

            # Running_mean_t = running_mean_t-1 + 1/t * (Gyx - running_mean_t-1)
            running_estimate = 0

            # Max_restarts: how often do we restart the coupled chains? 
            max_restarts = max_restarts
            # Keeping track of the number of restarted couplings
            num_restarts = 0
            
            for i in range(max_restarts):
                # Initializing state Y if it is random
                if initial_state_Y == "random":
                    # Corresponds to page 6 in the original paper: fishy function for g_{\nu}
                    i = np.random.randint(low = 0,high=len(self.env_Y.S))
                    state_Y = self.env_Y.S[i]
                else:
                    state_Y = initial_state_Y
                    
                # Initializing our estimator for a single run (Gyx)
                Gyx = 0

                # Reset the environment to the initial states
                state_X = tuple(self.env_X.reset((initial_state_X)))
                state_Y = tuple(self.env_Y.reset((state_Y)))
                
                # When one of the agents reaches the target, the episode is over (not in continuing environments)
                terminated_X, terminated_Y = False, False

                while (not terminated_X) and (not terminated_Y) and (state_X != state_Y):

                    if coupling == "maximal":
                        # The episode terminates when X and Y meet or, if not continuing, when one of both reaches the target
                        (new_X, new_Y) = self.sample_coupled_states(state_X,state_Y)
                        action_X = next_state_to_action(state_X, new_X)
                        action_Y = next_state_to_action(state_Y, new_Y)
                    elif coupling == "independent":
                        # select actions according to an independent coupling
                        action_X = self.select_action(state_X)
                        action_Y = self.select_action(state_Y)

                    # Take a step in both environments
                    state_X, reward_X, terminated_X, truncated_X, info_X = self.env_X.step(action_X)
                    state_Y, reward_Y, terminated_Y, truncated_Y, info_Y = self.env_Y.step(action_Y)
                    # get obs in gridworld.py returns a list, we need a tuple for indexing
                    # only in case the new_X or new_Y is the target, we receive the reward but do not go to new_X 
                    # or new_Y, instead we teleport to the starting state
                    state_X, state_Y = tuple(state_X), tuple(state_Y)
                    Gyx += reward_X - reward_Y
                    

                
                # The average total difference of a single run, averaged over multiple restarts
                num_restarts += 1
                # Incremental update of the mean
                running_estimate = running_estimate + (1/num_restarts) * (Gyx - running_estimate)
                estimates.append(running_estimate)

                # Early stopping criterion
                if (num_restarts>2) and (abs(estimates[-2] - estimates[-1]) < epsilon):
                    break

            if plot_convergence:
                axs.plot(estimates)
                axs.set_xlabel("Number of restarts of the couplings")
                axs.set_title(f"Estimated bias for state {initial_state_X}, where y = {initial_state_Y}")
                plt.show()
            
            if not tol_interval_computation:
                print(f"For initial state {initial_state_X}, we find {running_estimate}")
            # If we want to compute the tolerance interval, we are only interested in the estimates over time
            if tol_interval_computation:
                return estimates
            
            Gy.loc[[initial_state_X]] = running_estimate
        return Gy
            
    
    def simulate_coupled_lagged_chains(self, initial_X, initial_Y,L, coupling = "maximal"):
        """
        Simulates two coupled lagged chains.

        Simulates two chains, X and Y, that have been coupled. The chains evolve based on the
        given initial states and lag parameter.

        Parameters:
        - initial_X (str or object): Initial state for chain X. If "random", a random state
        is selected. Otherwise, provide the initial state as a tuple.
        - initial_Y (str or object): Initial state for chain Y. If "random", a random state
        is selected. Otherwise, provide the initial state as a tuple.
        - L (int): The lag parameter.
        - coupling (string): maximal or independent coupling

        Returns:
        - chain_X (list): List containing states of chain X at each time step, including the
        initial state.
        - chain_Y (list): List containing states of chain Y at each time step, including the
        initial state.
        - tau (int): meeting time of the chains (number of elements in Y)

        Notes:
        - If initial_X and initial_Y are both set to "random," random initial states are
        selected independently from their respective state spaces.
        - The chains are simulated by iteratively sampling new states based on the previous
        states, and they are coupled until their states become equal.

        Example:
        ```
        initial_X = "random"
        initial_Y = "random"
        L = 10
        coupling = "maximal"
    
        chain_X, chain_Y, tau = simulate_coupled_lagged_chains(initial_X, initial_Y, L, coupling)
        ```
        """
        # Initializing state X and Y, if random, both X0 and Y0 follow \mu

        
        if initial_X == initial_Y == "random":
            # Corresponds to page 6 in the original paper: fishy function for g_{\nu}
            i = np.random.randint(low = 0,high=len(self.env_X.S))
            state_X = self.env_X.S[i]

            j = np.random.randint(low = 0,high=len(self.env_Y.S))
            state_Y = self.env_Y.S[j]
        else:
            state_X = initial_X
            state_Y = initial_Y

        # Storing and appending the chains
        chain_X = [state_X]
        chain_Y = [state_Y]

        for lag in range(L):
            # Sample a new state based on the previous state
            chain_X.append(self.sample_single_state(chain_X[-1]))

        met = False
        # Keep track of the meeting time
        tau = 0

        while not met: 
            if coupling == "maximal":
                (newX, newY) = self.sample_coupled_states(chain_X[-1], chain_Y[-1])
            elif coupling == "independent":
                newX = self.sample_single_state(chain_X[-1])
                newY = self.sample_single_state(chain_Y[-1])
            chain_X.append(newX)
            chain_Y.append(newY)
            tau += 1
            if newX == newY:
                met = True

        
        return chain_X, chain_Y, tau
    
    def select_action(self, state):
        """
        Select an action according to the policy
        """
        dp = self.policy.loc[[state]]

        def rp():
            return np.random.choice(dp.columns, p = np.float16(dp.values[0]))
        
        action = rp()
        return action
        


    def sample_single_state(self, state_X):
        """
        Sample state for agent X according to kernel K(x_{t-1}, . ), used to sample X_1,...,X_L
        """
        #### Defining the kernel to sample from, to find the density from and the target from which we want to 
        #### evaluate the density

        if state_X == tuple(self.env_X._target_location):
            # When we reach the target, we immediately go to the start position in the bottom left
            return (0,0)

        dp = self.K.loc[[state_X]]

        def rp():
            return np.random.choice(dp.columns, p = np.float16(dp.values[0]))
        
        new_X = rp()
        
        return new_X
    



    def sample_coupled_states(self, state_X, state_Y):
        """
        This function will sample coupled states for X and Y given the previous states in the chains. 
        """

        ###########
        #  Gamma coupling: maximally couple K(x->.) and K(y->.)
        ###########
        (new_X, new_Y) = self.gamma_coupling(state_X, state_Y)

        # If we reach the target, we go back to start
        if state_X == tuple(self.env_X._target_location):
            # When we reach the target, we immediately go to the start position in the bottom left
            new_X =  (0,0)
        
        if state_Y == tuple(self.env_Y._target_location):
            # When we reach the target, we immediately go to the start position in the bottom left
            new_Y = (0,0)
        return (new_X, new_Y)


    def gamma_coupling(self, state_X, state_Y) -> np.array:
        """
        Function that samples from the gamma-coupling (so called by Jensen)

        Params: 
            state_X: the current state of Markov Chain X
            state_Y: the current state of Markov Chain Y

        Created variables: 
            dp (callable): evaluate the density of the kernel K: K(x->.)
            dq (callable): evaluate the density of the kernel K: K(y->.)
            rp: draw a random sample from kernel K: K(x->.)
            rq: draw a random sample from kernel K: K(y->.)
        """

        ###########
        # Setting up the densities and functions for sampling
        ###########

        # Evaluating the densities of X and Y: these are pandas dataframes with n_states columns giving P(state_x -> .)
        # evaluation happens with dp[new_state].values[0]
        dp = self.K.loc[[state_X]]
        dq = self.K.loc[[state_Y]]

        # Sampling from X and Y
        def rp():
            return np.random.choice(dp.columns, p = np.float32(dp.values[0]))
        
        def rq():
            return np.random.choice(dq.columns, p =np.float32(dq.values[0]))

        ###########
        # Performing the gamma coupling
        ###########
        X = rp()
        W = np.random.uniform()

        if W <= dq[X].values[0]/dp[X].values[0]:
            Y = X
        else: 
            reject = True
            while reject:
                # Loop executed at least once, when W larger than the acceptance_ratio, 
                Y = rq()
                W = np.random.uniform()
                reject = (W <= dp[Y].values[0]/dq[Y].values[0])
        return (X,Y)
    

    def show_policy(self):
        state = (0,0)
        self.env_X.reset(initial_location=state)

        self.env_X.render_plot()
        terminated = False
        i=0
        max_iter = 25
        while (not terminated) and (i<max_iter):
            action = np.random.choice(list(range(len(self.env_X.A))), p = self.policy.loc[[state]].values[0])
            state, reward, terminated, truncated, info = self.env_X.step(action)
            self.env_X.render_plot()
            i+=1


    def run_Poisson_RL(self, plot_convergence = False, max_restarts = 200, epsilon = np.inf, initial_state_Y = (0,0)):
        """
        Running the whole algorithm as a control algorithm: evaluating a policy and then greedifying
        It returns a policy as well, this is what we call from outside of this class, that is why it has some redundant parameters
        with the evaluate policy method, which is the main method of our class. 
        """
        num_iters = 1
        for i in range(num_iters):
            self.Gy = self.evaluate_policy(plot_convergence, max_restarts, epsilon, initial_state_Y)
            self.policy = greedify_policy(self.Gy,self.env_X,self.env_Y, self.policy)

        return self.Gy, self.policy
    
    def plot_tolerance_interval(self, eval_state, max_restarts, num_runs):
        """
        Function to plot the tolerance interval of the estimates for the value of a state
        """
        all_estimates = []
        for run in tqdm(range(num_runs)):
            estimates_run = self.value_estimation(plot_convergence=False,epsilon = 0.0,max_restarts=max_restarts,coupling="independent", tol_interval_computation=True, evaluation_state=eval_state)
            all_estimates.append(estimates_run)

        all_estimates = np.array(all_estimates)
        means = np.mean(all_estimates, axis=0)
        # Assuming normal distribution as underlying distribution
        bounds = ti.twoside.normal(all_estimates.transpose(), 0.9, 0.95)
        lower_bounds_ti = bounds[:,0]
        upper_bounds_ti = bounds[:,1]

        plt.plot(np.arange(len(means)), means, color='black', label=f'Mean over {num_runs} runs. Final estimate of {round(means[-1],2)}')

        # Fill the area between lower and upper bounds
        plt.fill_between(np.arange(len(means)), lower_bounds_ti, upper_bounds_ti, color='gray', alpha=0.5, label='Tolerance Interval')

        # Customize the plot
        plt.xlabel('Restarts')
        plt.ylabel('Value Estimate')
        plt.title(f'Tolerance interval for the value of state {eval_state}.')
        plt.legend()
        # Adjust ylim on plot, shows drastic tolerance interval after only very few restarts
        plt.ylim([lower_bounds_ti[20], upper_bounds_ti[20]])
        plt.grid(True)
        # Show the plot
        plt.show()

    def gain_tol_interval(self, max_restarts ,num_runs, average_reward_long_run, L, k,l):
        """
        Plot tolerance intervals for the estimation of the gain

        Params: 
            max_restarts: the number of simulated couplings in a single run
            num_runs: the number of independent runs to compute tolerance intervals
        """
        env_X = Gridworld(size=7, continuing = True, stochastic=True)
        env_Y = Gridworld(size=7, continuing = True, stochastic=True)

        all_estimates = []

        for run in tqdm(range(num_runs)):
            estimates = []
            running_estimate=0
            for num_restarts in range(1,max_restarts):
                Hkl, _, cost = self.sample_unbiased_estimator(L=L,k=k,l=l)
                running_estimate = running_estimate + (1/num_restarts) * (Hkl - running_estimate)
                estimates.append(running_estimate)

            all_estimates.append(estimates)

        all_estimates = np.array(all_estimates)
        means = np.mean(all_estimates, axis=0)
        # Assuming normal distribution as underlying distribution
        bounds = ti.twoside.normal(all_estimates.transpose(), 0.9, 0.95)
        lower_bounds_ti = bounds[:,0]
        upper_bounds_ti = bounds[:,1]

        plt.plot(np.arange(len(means)), means, color='black', label=f'Mean over {num_runs} runs. Final estimate of {round(means[-1],4)}')

        # Fill the area between lower and upper bounds
        plt.fill_between(np.arange(len(means)), lower_bounds_ti, upper_bounds_ti, color='gray', alpha=0.5, label='Tolerance Interval')

        # Customize the plot
        plt.xlabel('Restarts')
        plt.ylabel('Gain Estimate')
        plt.title(f'Tolerance interval for gain estimation.')
        plt.hlines(average_reward_long_run, xmin=0,xmax=num_restarts, linestyles='--', colors='gold')
        plt.legend()
        # Adjust ylim on plot, shows drastic tolerance interval after only very few restarts
        plt.ylim([lower_bounds_ti[20], upper_bounds_ti[20]])
        plt.grid(True)
        # Show the plot
        plt.show()
                    


        
    def sample_unbiased_estimator(self,k,l,L, X_0 = "random", Y_0 = "random"):
        if X_0 == Y_0 == "random":
            # Corresponds to page 6 in the original paper: fishy function for g_{\nu}
            i = np.random.randint(low = 0,high=len(self.env_X.S))
            X_0 = self.env_X.S[i]

            j = np.random.randint(low = 0,high=len(self.env_Y.S))
            Y_0 = self.env_Y.S[j]
        else:
            X_t = X_0
            Y_t = Y_0

        X_t = tuple(self.env_X.reset((X_0)))
        Y_t = tuple(self.env_Y.reset((Y_0)))

        time = 0
        cost = 0
        mcmcestimator = 0
        bias_correction = 0

        for t in range(1,L+1):
            # The action at time t-1 depends on the state at time t-1
            X_a_t = self.select_action(X_t)
            time += 1
            X_t, X_r_t, _, _, _ = self.env_X.step(X_a_t)
            X_t = tuple(X_t)
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
                X_a_t, Y_a_t = self.select_action(X_t), self.select_action(Y_t)
                X_t, X_r_t, _, _, _ = self.env_X.step(X_a_t)
                X_t = tuple(X_t)
                Y_t, Y_r_t, _, _, _ = self.env_Y.step(Y_a_t)
                Y_t = tuple(Y_t)
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
                X_a_t = self.select_action(X_t)
                X_t, X_r_t, _, _, _ = self.env_X.step(X_a_t)
                X_t = tuple(X_t)
                cost += 1
                if (k<=time<=l):
                    mcmcestimator += X_r_t
                
        unbiased_estimator = (mcmcestimator + bias_correction)/(l-k+1)

        return unbiased_estimator, meeting_time, cost
        
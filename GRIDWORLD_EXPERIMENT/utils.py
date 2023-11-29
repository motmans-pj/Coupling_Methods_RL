import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

def state_transition_kernel(size):
    '''
    Return the state transition kernel for a gridworld environment of given size
    P(S_t+1 = s' |S_t = s, A_t = a)

    Params: 
        size (int): what is the size of the gridworld

    Returns: 
        state transition kernel: a pandas dataframe with multi-indices
    '''

    # A list of the 36 possible states
    states = list(range(size))
    states = [(i, j) for i in states for j in states]

    actions = list(range(5))
    indices = [states, actions]
    # Create a multi-index from the states and actions, first level are the states, second level are the actions
    P = pd.MultiIndex.from_product(indices, names = ["States", "Actions"])

    # State (0,0)
    ### Action 0 => transition kernel goes to (0,0) with probability 1
    ### Action 1 => go to (1,0) with probability 1

    # Now fill a dataframe of zeros with this multi index
    df = pd.DataFrame(np.zeros((size**2 * len(actions), size**2)), index=P)
    df.columns = states

    for s in states:
        for a in actions:

            # Staying in place
            if a == 0:
                ######  df.xs(old_state, level="States").loc[action].at[new_state] = 1
                # xs(old_state) selects the dataframe for the old_state (1 row for each action)
                # loc[action] selects the row corresponding to that action
                # at[new_state] helps us change the value in that row for that state
                
                # Sure to transition to same state when standing still
                df.xs(s, level = "States").loc[a].at[s] = 1

            # Moving right
            if a == 1:
                new_state = (s[0]+1, s[1])
                if new_state in states:
                    df.xs(s, level = "States").loc[a].at[new_state] = 1
                else: 
                    df.xs(s, level = "States").loc[a].at[s] = 1

            # Moving up
            if a == 2:
                new_state = (s[0], s[1]+1)
                if new_state in states:
                    df.xs(s, level = "States").loc[a].at[new_state] = 1
                else: 
                    df.xs(s, level = "States").loc[a].at[s] = 1

            # Moving left
            if a == 3:
                new_state = (s[0]-1, s[1])
                if new_state in states:
                    df.xs(s, level = "States").loc[a].at[new_state] = 1
                else: 
                    df.xs(s, level = "States").loc[a].at[s] = 1

            # Moving down
            if a == 4:
                new_state = (s[0], s[1]-1)
                if new_state in states:
                    df.xs(s, level = "States").loc[a].at[new_state] = 1
                else: 
                    df.xs(s, level = "States").loc[a].at[s] = 1
            
    return df


def initialize_policy(states, actions, kind = "Uniform"):
    """
    Initializing a policy that for each state, defines a probability distribution over the actions pi(a|s)

    Params: 
        states: the states of the environment
        actions: the actions in the environment
        kind: we can initialize the policy differently
    """

    if kind == "Uniform":
        # Each action is equally likely in each state
        # returns a dataframe of size (num_states, num_actions)
        return pd.DataFrame( (1/len(actions)) * np.ones((len(states), len(actions))), index = states, columns = actions)

    # We can also extend this to include initialization from a normal distribution, 
    # and then divide each number by the sum so the sum adds up to 1


def MC_transition_matrix(P, pi):
    """
    Computes the Markov Chain transition kernel corresponding to a certain policy (average reward RL paper)
    
    Params: 
        P: the state transition kernel corresponding to the dynamics of the environment
        it is a multi-index matrix with the old state and action as indices, and the next state as columns. 
        Entries correspond to p(s'|s,a). 
        pi: the policy from which the MC comes

    Returns: 
        The transition matrix for the MC, K[S,S'] = the probability of transitioning from S to S'
        It can be computed as \sum_{a \in A} pi(a|s) * p(s' | s,a)

        It will return a dataframe of size (len(states), len(states))
    """

    K = pd.DataFrame(index=pi.index, columns = pi.index)

    for old_state in pi.index:
        for new_state in pi.index:
            transition_prob = 0
            for action in pi.columns:
                # pi[old_state, action] = \pi(a|old_state)
                # df.xs(old_state, level="States").loc[action].at[new_state] = 1
                transition_prob += pi.at[old_state, action] * P.xs(old_state, level="States").loc[action].at[new_state]
            K.at[old_state,new_state] = transition_prob

    return K



def argmax(q_values):
    """
    Takes in a list of q_values and returns the index
    of the item with the highest value. Breaks ties randomly. (numpy just selects the first maximal index)

    Follows page 79 of Sutton & Barto

    returns: the index of the highest value in q_values
    """
    top = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value, then update top and reset ties to zero
        # if a value is equal to top value, then add the index to ties 
        if q_values[i] > top:
           top = q_values[i]
           # reset ties
           ties = [i]
        # If they are equal up to some rounding error
        elif math.isclose(q_values[i], top, rel_tol = 1e-4):
           ties.append(i)
    # return a random selection from ties. 
    # We return all indices (actions) that maximize the value and give them equal probability
    # We can do this because the policy improvement theorem holds also for stochastic policies
    # The policy improvement states that a one-step lookahead for the value function is enough for 
    # all future actions, and that a greedy policy will always be non-strictly better
    return ties


# Remains in the file, but not used, our method is just a prediction algorithm that does not try to find the best policy 
def greedify_policy(Gy, env_X, env_Y, policy):        
    """
    Greedify the policy with respect to the obtained value function for the previous policy
    \pi{s} = \argmax_{a \in A} \mathbb{E} R_{t+1} + \gamma v_{\pi}(S_{t+1}|S_t = s, A_t = a)
    Which can be implemented with the following function

    Params: 
-        Gy: the value function of the policy 
         env_X: the environment for state X
         env_Y: the environment for state Y
    """

    # Create an dataframe of zeros in which we will store the new policy
    new_policy = pd.DataFrame(np.zeros((len(env_X.S), len(env_X.A))), index = policy.index, columns = policy.columns)
    for old_state in env_X.S:

        q_values = []
        for a in env_X.A:
            if a != 0:
                # We do not want to stand still when solving the environment, 
                # the action of doing nothing was only necessary for the coupling
                # and having an MCMC that does nothing
                q_values.append(np.sum([
                    env_X.state_transition_kernel.xs(old_state, level="States").loc[a].at[new_state] * 
                    (
                        env_X._get_reward(new_state)
                        +  Gy.loc[[new_state]].values[0][0]
                    )
                    for new_state in env_X.S
                    ]))
            else: 
                q_values.append(-np.inf)
                
        optimal_actions = argmax(q_values)
        for optimal_action in optimal_actions:
            
            # For each optimal action give it a probability of being chosen as 1/number of optimal actions
            new_policy.at[old_state, optimal_action] = 1/len(optimal_actions)
        
    return new_policy 



def next_state_to_action(old_state, next_state):
    """
    This function is a helper that says which action should be taken to go from one state to another
    This is only possible in a deterministic environment

    Params:
        old_state: state the agent finds itself in
        next_state: state that is the output of the MRTH algorithm, and is the next state in the MC kernel
                    to find out what reward is obtained in that next state, we need to do an env.step()
                    to get the reward, as such, we need an action that the algorithm prescribes
    """
    # Doing nothing
    if old_state == next_state:
        action = 0
    # Right
    elif (old_state[0]+1) == next_state[0]:
        action = 1
    # Up
    elif (old_state[1]+1) == next_state[1]:
        action = 2

    # Left
    elif (old_state[0]-1) == next_state[0]:
        action = 3

    # Down
    elif (old_state[1]-1) == next_state[1]:
        action = 4
    # If the next state is not any of these, we reset from the target back to the start, so we can take any action
    else: 
        action = 0

    return action

def render_plot_both(env_X, env_Y):
    """
    Showing both agents on a single plot
    """
    plt.figure(figsize=(3, 3))
    plt.xlim(-0.5,  env_X.size-0.5)
    plt.ylim(-0.5,  env_X.size-0.5)
    plt.xticks(np.arange(0.5,  env_X.size, step=1))
    plt.yticks(np.arange(0.5,  env_X.size, step=1))
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    plt.grid()

    plt.plot( env_X._agent_location[0],  env_X._agent_location[1], 'ro', markersize=15, label = 'Agent X')
    plt.plot( env_Y._agent_location[0],  env_Y._agent_location[1], 'go', markersize=15, label = 'Agent Y')

    plt.plot( env_X._target_location[0],  env_X._target_location[1], 'bs', markersize=15, label = 'Target')
    plt.legend()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
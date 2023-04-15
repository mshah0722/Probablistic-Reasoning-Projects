import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    # Initializing variables
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    for step in range(num_time_steps):
        forward_messages[step] = rover.Distribution()
        backward_messages[step] = rover.Distribution()
        marginals[step] = rover.Distribution()
    
    # TODO: Compute the forward messages
    # Compute the first forward message (alpha(z0))
    for z in prior_distribution:
        forward_messages[0][z] = prior_distribution[z] * observation_model(z)[observations[0]]
    
    forward_messages[0].renormalize()
 
    # Using recursion compute the rest of the forward messages
    for step in range(num_time_steps - 1):
        curr_observation = observations[step + 1]
        
        for state in all_possible_hidden_states:            
            if curr_observation == None:
                prob_obs = 1
            
            else:
                prob_obs = observation_model(state)[curr_observation]
            
            curr_sum = sum(forward_messages[step][last_state] * transition_model(last_state)[state] for last_state in forward_messages[step]) 
            forward_messages[step + 1][state] = prob_obs * curr_sum
        
        # Renormalize
        forward_messages[step + 1].renormalize()   
                   
    # TODO: Compute the backward messages
    # Compute the first backward message (beta(z(n-1)))
    beta_zn1 = rover.Distribution()
    
    for state in all_possible_hidden_states:
        beta_zn1[state] = 1
    
    beta_zn1.renormalize()
    
    backward_messages[num_time_steps - 1] = beta_zn1
    
    # Using recursion compute the rest of the backward messages
    for step in reversed(range(num_time_steps - 1)):
        curr_observation = observations[step + 1]
        
        for state in all_possible_hidden_states:
            prob_trans = transition_model(state)
            
            if curr_observation == None:
                backward_messages[step][state] = sum(backward_messages[step + 1][next_state] * prob_trans[next_state] for next_state in backward_messages[step + 1])
            
            else:
                backward_messages[step][state] = sum(backward_messages[step + 1][next_state] * prob_trans[next_state] * observation_model(next_state)[curr_observation] for next_state in backward_messages[step + 1])
                
        # Renormalize
        backward_messages[step].renormalize()
    
    # TODO: Compute the marginals 
    for step in range(num_time_steps):        
        for state in all_possible_hidden_states:
            alpha_zi = forward_messages[step][state]
            beta_zi = backward_messages[step][state]
            marginals[step][state] = alpha_zi * beta_zi
        
        # Renormalize
        marginals[step].renormalize()
        
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    # Initializing variables
    time_step_count = len(observations)
    w = [None] * time_step_count
    max_states = [None] * time_step_count
    estimated_hidden_states = [None] * time_step_count
    
    # Compute the first omega value (w(0))
    w[0] = rover.Distribution()
    
    for state in prior_distribution:
        curr_pd = prior_distribution[state]
        curr_observation = observations[0]
        
        if curr_observation == None:
            prob_obs = 1
        
        else:
            prob_obs = observation_model(state)[curr_observation]
            
        if(curr_pd * prob_obs) != 0:
            w[0][state] = np.log(curr_pd * prob_obs)
    
    # Using recursion compute the rest of the omega values
    for step in range(time_step_count - 1):
        w[step + 1] = dict()
        max_states[step + 1] = dict()
        curr_observation = observations[step + 1]
        
        for state in all_possible_hidden_states:
            max_w = -float('inf')
            max_state = None
            
            if curr_observation == None:
                prob_obs = 1
        
            else:
                prob_obs = observation_model(state)[curr_observation]
                
            calculation = {last_state:(np.log(transition_model(last_state)[state]) + w[step][last_state]) for last_state in w[step].keys()}
            max_w = max(calculation.values()) 
            max_state = max(calculation, key=calculation.get)
            max_states[step + 1][state] = max_state
                
            if prob_obs != 0:
                w[step + 1][state] = max_w + np.log(prob_obs)
                
            else:
                w[step + 1][state] = np.log(prob_obs)
    
    # Find the best state at the last time step
    estimated_hidden_states[time_step_count - 1] = max(w[time_step_count - 1], key=w[time_step_count - 1].get)
    
    # Backtracking to find the sequence of max likelihood
    for step in reversed(range(num_time_steps - 1)):
        estimated_hidden_states[step] = max_states[step + 1][estimated_hidden_states[step + 1]]
        
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
        
    print('\n')
    
    # initializing the varibales
    for_back_error = 0
    for_back_most_likely = [None] * num_time_steps
    viterbi_error = 0
    
    # Calculating the most likely state at each timestep using the forward-backward algorithm
    for time_step in range(num_time_steps):
        max_prob = 0
        max_state = None
        
        for state in marginals[time_step]:
            if marginals[time_step][state] > max_prob:
                max_prob = marginals[time_step][state]
                max_state = state
                
        for_back_most_likely[time_step] = max_state
        
        # Calculating the error for the forward-backward algorithm
        if for_back_most_likely[time_step] != hidden_states[time_step]:
            for_back_error += 1
        
        # Calculating the error for the Viterbi algorithm
        if estimated_states[time_step] != hidden_states[time_step]:
            viterbi_error += 1
            
    # Calculating the error rate for the forward-backward algorithm
    for_back_error_rate = for_back_error / num_time_steps
    
    # Calculating the error rate for the Viterbi algorithm
    viterbi_error_rate = viterbi_error / num_time_steps
    
    print("Error rate for the forward-backward algorithm: %f" % (for_back_error_rate))
    print("Error rate for the Viterbi algorithm: %f" % (viterbi_error_rate))
      
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
        
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        

import logging

from mdp import GridMDP, MDP, value_iteration, policy_evaluation
from utils import turn_left, turn_right
from optparse import OptionParser
from random import randint
from time import time
from itertools import product

class GridMDP(GridMDP):

    char_switch = {
        '>' : (1,0),
        '^' : (0,1),
        '<' : (-1, 0),
        '.' : None
    }

    # TODO: this and the next should be static methods
    def char_to_tuple(self, direction):
        return self.char_switch[direction]

    def tuple_to_char(self, tuple):
        for k,v in self.char_switch.items():
            if v == tuple:
                return k

        return None

    def simulate_move(self, state, action):
        # TODO: get percentages from T
        random_number = randint(0, 100)
        if (random_number >= 0) and (random_number <= 9):
            return self.go(state, turn_right(action))
        elif (random_number >= 10) and (random_number <= 20):
            return self.go(state, turn_left(action))
        else:
            return self.go(state, action)


class MyMDP(MDP):
    """ Extends MDP class to use a dictionary transistion model """
    
    def __init__(self, init, actlist, terminals, gamma=.9):
        MDP.__init__(self,init, actlist, terminals, gamma)
        self.model = { }

    def R(self, state): 
        " Return a numeric reward for this state. "
        if state in self.reward:
            return self.reward[state]
        else:
            # TODO: this should really return zero? or return False beause we
            # don't know.  Returns 0 for now as it makes the value iteration
            # function work
            return 0
            #raise Exception('tried to get reward of state we dont have yet %s' % str(state))

    def T(self, state, action):
        " Returns a list of tuples with probabilities for states "
        try:
            possible_results_and_probabilities = self.model[state][action]
        except KeyError:
            return []
        
        l = []
        for result_state, probability in possible_results_and_probabilities.items():
            l.append((probability, result_state))
        return l
    
    def T_add(self, state, action, result_state, probability):
        " Adds a value to the transistion model "
        if (state in self.model) and (action in self.model[state]):
            self.model[state][action][result_state] = probability
        elif (state in self.model):
            self.model[state][action] = { result_state : probability }
        else:
            self.model[state] = {action : { result_state : probability} }

class PassiveADPAgent(object):

    def __init__(self, action_mdp, policy):
        self.mdp = MyMDP(init=(0, 0),
                       actlist=[(1,0), (0, 1), (-1, 0), (0, -1)],
                       terminals=action_mdp.terminals,
                       gamma = 0.9)
        self.action_mdp = action_mdp
        self.utility, self.outcome_freq = { }, { }
        self.reached_states = set([])
        self.previous_state, self.previous_action = None, None
        self.create_policy_and_states(policy)
        self.create_empty_sa_freq()

    def create_empty_sa_freq(self):
        " Creates state action frequences with inital values of 0 "
        self.sa_freq = { }
        for state in self.mdp.states:
            self.sa_freq[state] = { }
            for action in self.mdp.actlist:
                self.sa_freq[state][action] = 0.0
        
    def create_policy_and_states(self, policy):
        " Sets the initial policy, and also sets the mdp's states "
        self.policy = {}
        self.mdp.states = set()

        ## Reverse because we want row 0 on bottom, not on top
        policy.reverse() 
        self.rows, self.cols = len(policy), len(policy[0])
        for x in range(self.cols):
            for y in range(self.rows):
                # Convert arrows to numbers
                if policy[y][x] == None:
                    self.policy[x, y] = None
                else:
                    self.policy[x, y] = self.action_mdp.char_to_tuple(policy[y][x])

                # States are all non-none values
                if policy[y][x] is not None:
                    self.mdp.states.add((x, y))

    def add_state_action_pair_frequency(self, state, action):
        self.sa_freq[state][action] += 1
    
    def get_state_action_pair_frequency(self, state, action):
        return self.sa_freq[state][action]
        
    def add_outcome_frequency(self, state, action, outcome):
        # We haven't seen this state yet
        if state not in self.outcome_freq:
            self.outcome_freq[state] = {action : {outcome : 1}}
            return
        
        # We've seen the state but not the action
        if action not in self.outcome_freq[state]:
            self.outcome_freq[state][action] = {outcome : 1}
            return
        
        # We've seen the state and the action, but not the outcome
        if outcome not in self.outcome_freq[state][action]:
            self.outcome_freq[state][action][outcome] = 1
            return
        
        # We've seen the state, action, and outcome, add 1
        self.outcome_freq[state][action][outcome] += 1
    
    def get_outcome_frequency(self, state, action, outcome):
        try:
            return self.outcome_freq[state][action][outcome]
        except KeyError:
            return 0

    def print_outcome_frequency(self):
        for state in agent.outcome_freq:
            for action in agent.outcome_freq[state]:
                for result_state, result_frequency in agent.outcome_freq[state][action].items():
                    print 'state', state, '\t action', action, \
                    '\t result state',result_state, '\t frequency', result_frequency


    def get_move_from_policy(self, state_x, state_y):
        return self.policy[state_x][state_y]
        
    def next_action(self, current_state, current_reward):
        # policy = self.policy computed by constructor
        # MDP = mdp object. self.mdp
        #          MDP.T  - transistion model (initially empty),
        #          MDP.reward - reward
        #          MDP gamma in initializer
        # utility = dictionary [(0,0)] = 0.57 etc
        # state action frequencies = sa_freq (dict) initially empty
        # outcome frequencies given state outcome and state-action pairs = outcome_freq  initially empty
        #     dict with key being new state, value being another dict with keys being
        #     state, action pairs and values being that percentage
        # previous state, previous action = s,a
        
        # if s' is new then:
        if (current_state not in self.reached_states):
            # U[s'] <- r'
            self.utility[current_state] = current_reward
            
            # R[s'] <- r'
            self.mdp.reward[current_state] = current_reward
            
            # Make sure we know we have seen it before
            self.reached_states.add(current_state)
        
        # if s is not null
        if self.previous_state is not None:
            # increment Nsa[s,a] and Ns'|sa[s', s, a]
            self.add_state_action_pair_frequency(self.previous_state, self.previous_action)
            self.add_outcome_frequency(self.previous_state, self.previous_action, current_state)
            
            # for each t such that Ns'|sa[t,s,a] is nonzero:
            for state in agent.outcome_freq:
                for action in agent.outcome_freq[state]:
                    for result_state, result_frequency in agent.outcome_freq[state][action].items():
                        if result_frequency > 0:
                            # P (t, s, a) <- Ns'|sa[t, s, a] / Nsa[s,a]
                            # Update the model to be:
                            # ((freq of this action happening with this state action pair)
                            # / (total freq of this state action pair combo))
                            probability = result_frequency / self.get_state_action_pair_frequency(state, action)
                            self.mdp.T_add(state, action, result_state, probability)
        
        self.utility = policy_evaluation(self.policy, self.utility, self.mdp)

        # if s'.TERMINAL?
        # If we're at a terminal we don't want a next move
        if current_state in self.mdp.terminals:
            logging.info('Reached terminal state %s' % str(current_state))
            # s,a <- null
            self.previous_state, self.previous_action = None, None
            return False
        else:
            # s,a <- s', policy[s']
            next_action = self.policy[current_state]
            self.previous_state, self.previous_action = current_state, next_action
            # Return the next action that the policy dictates
            return next_action

    
    def execute_trial(self):
        # Start at initial state
        current_state = self.mdp.init
        
        # Keep going until we get to a terminal state
        while True:
            logging.info('--------------------------')

            # Get reward for current state
            current_reward = self.action_mdp.R(current_state)

            # Calculate move from current state
            next_action = self.next_action(current_state, current_reward)

            logging.info('Current State: %s ' % str(current_state))
            logging.info('Current Reward: %s ' % current_reward)
            logging.info('Next action: %s' % self.action_mdp.tuple_to_char(next_action))

            if next_action == False:
                # End because next_action told us to
                logging.info('Next_action returned false, stopping')
                break

            # Get new current_state
            current_state = self.action_mdp.simulate_move(current_state, next_action)

if __name__ == '__main__':
    ''' Parses options from command line, creates Fig 17,1, runs the passive
    adp agent on it certain amount of times, outputs info and utilities '''
    
    # Setup file options
    parser = OptionParser()
    parser.add_option("-t", "--times", dest="times", type="int", default = 100,
                      help="times to run")
    parser.add_option("-d", "--debug", action='store_true', dest="debug",
                      default=False, help="debug mode?")
    parser.add_option("-i", "--info", action='store_true', dest="info",
                      default=False, help="info mode?")
    parser.add_option("-f", "--file", dest="log_file",
                      default=False, help="file to log to")
    (options, args) = parser.parse_args()
    
    if options.debug:
        level = logging.DEBUG
    elif options.info:
        level = logging.INFO
    else:
        level = logging.CRITICAL
    
    format = '%(levelname)s: %(message)s'
    if options.log_file:
        logging.basicConfig(level=level,
                            filename=options.log_file,
                            filemode='w',
                            format=format)
    else:
        logging.basicConfig(level=level,
                            format=format)
    
    # Set up grid MDP to act on
    Fig = {}
    Fig[17,1] = GridMDP([[-0.04, -0.04, -0.04, +1.0],
                         [-0.04, None, -0.04, -1.0],
                         [-0.04, -0.04, -0.04, -0.04]],
                        terminals=[(3, 2), (3, 1)])
    
    # Setup values
    policy = [['>', '>', '>', '.'],
              ['^', None, '^', '.'],
              ['^', '<', '<', '<']]
    
    # Create agent
    agent = PassiveADPAgent(Fig[17,1], policy)
    
    # Start timing
    time_start = time()
    logging.info('Start at %s' % time_start)
    
    # Execute a bunch of trials
    trials = options.times
    for i in range (0,trials):
        agent.execute_trial()
    
    # End timing
    time_end = time()
    logging.info('End at %s' % time_end)
    
    seconds_elapsed = time_end - time_start
    minutes_elapsed = seconds_elapsed / 60.0
    
    # Print and log final results
    final_results = (('Took %d seconds, which is %d minutes' % (seconds_elapsed, minutes_elapsed)),\
        ('Executed %i trials' % (trials)), ('Utilities: %s' % (agent.utility)))
    for result in final_results:
        logging.info(result)
        print result

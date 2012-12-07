from mdp import GridMDP, MDP, value_iteration, policy_evaluation, Fig
from utils import update
from random import random
from time import time
import agents

class PassiveADPAgent(agents.Agent):
    """Passive (non-learning) agent that uses adaptive dynamic programming
    on a given MDP and policy. [Fig. 21.2]"""
    class LearntMDP:
        def __init__(self, states, gamma, terminals):
            update(self, P={}, reward={}, states=states, gamma=gamma, terminals=terminals)
            
        def R(self, s):
            """Return a numeric reward for the state s"""
            if s in self.reward:
                return self.reward[s]
            else:
                return 0. # we don't know the value of the reward.
            
        def T(self, s, a):
            """Returns a list of tuples with probabilities for states"""
            try:
                return [(p,s) for (s,p) in self.P[s][a].items()]
            except KeyError:
                return []
            
        def T_add(self, (s,a,t), p):
            " Adds a value to the transistion model "
            if (s in self.P) and (a in self.P[s]):
                self.P[s][a][t] = p
            elif (s in self.P):
                self.P[s][a] = {t:p}
            else:
                self.P[s] = {a:{t:p}}
    
    def __init__(self, action_mdp, pi):
        update(self,
               pi = pi,
               mdp = self.LearntMDP(action_mdp.states,action_mdp.gamma,action_mdp.terminals),
               action_mdp = action_mdp,
               U = {},
               Ns_sa = {s:{a:{t:0 for (p,t) in action_mdp.T(s,a)}
                           for a in action_mdp.actlist}
                        for s in action_mdp.states},
               Nsa = {s:{a:0. for a in action_mdp.actlist}
                      for s in action_mdp.states},
               s = None,
               a = None)
        
    def program(self, s1, r1):
        mdp,U,s,a,Nsa,Ns_sa = self.mdp,self.U,self.s,self.a,self.Nsa,self.Ns_sa
        if s1 not in mdp.reward: # mdp.R also tracks the visited states
            U[s1] = r1
            mdp.reward[s1] = r1
        if s is not None:
            Nsa[s][a] += 1
            Ns_sa[s][a][s1] += 1
            for t in Ns_sa[s][a]:
                if Ns_sa[s][a][t] > 0:
                    self.mdp.T_add((s,a,t), Ns_sa[s][a][t] / Nsa[s][a])
        U = policy_evaluation(self.pi, U, mdp)
        if s1 in mdp.terminals:
            self.s, self.a = None, None
            return False
        else:
            self.s, self.a = s1, self.pi[s1]
            return self.a

def simulate(mdp,(s,a)):
    r = random() # 0 <= r <= 1
    p,s1 = zip(*(mdp.T(s,a)))
    for i in range(len(p)):
        if sum(p[:i+1]) >= r:
            return s1[i]

def execute_trial(agent,mdp):
    current_state = mdp.init
    while True:
        current_reward = mdp.R(current_state)
        next_action = agent.program(current_state, current_reward)
        if next_action == False:
            break
        current_state = simulate(mdp,(current_state, next_action))

def demoPassiveADPAgent():
    print 'DEMO PassiveADPAgent'
    print '--------------------'
    # Setup values
    policy = {(0, 1): (0, 1),
              (1, 2): (1, 0),
              (3, 2): None,
              (0, 0): (0, 1),
              (3, 0): (-1, 0),
              (3, 1): None,
              (2, 1): (0, 1),
              (2, 0): (0, 1),
              (2, 2): (1, 0),
              (1, 0): (1, 0),
              (0, 2): (1, 0)}
    
    # Create agent
    time_start = time()
    trials = 100
    agent = PassiveADPAgent(Fig[17,1], policy)
    for i in range (0,trials):
        execute_trial(agent,Fig[17,1])
    time_end = time()
    
    seconds_elapsed = time_end - time_start
    minutes_elapsed = seconds_elapsed / 60.0
    final_results = (('Took %d seconds, which is %d minutes' % (seconds_elapsed, minutes_elapsed)),\
                     ('Executed %i trials' % (trials)), ('Utilities: %s' % (agent.U)))
    for result in final_results:
        print result

if __name__ == '__main__':
    demoPassiveADPAgent()

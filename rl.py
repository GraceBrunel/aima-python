"""Reinforcement Learning (Chapter 21)
"""

from mdp import value_iteration, policy_evaluation, policy_iteration, \
     GridMDP, MDP, Fig
from utils import update, argmax
from random import random
from time import time
from itertools import product
import agents

class PassiveADPAgent(agents.Agent):
    """Passive (non-learning) agent that uses adaptive dynamic programming
    on a given MDP and policy. [Fig. 21.2]"""
    class LearntMDP:
        """a model of the original mdp that the PassiveADP is trying to learn"""
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
            
        def T_set(self, (s,a,t), p):
            " Adds a value to the transistion model "
            if (s in self.P) and (a in self.P[s]):
                self.P[s][a][t] = p
            elif (s in self.P):
                self.P[s][a] = {t:p}
            else:
                self.P[s] = {a:{t:p}}
    
    def __init__(self, mdp, pi):
        update(self,
               pi = pi,
               mdp = self.LearntMDP(mdp.states,mdp.gamma,mdp.terminals),
               U = {},
               Ns_sa = {s:{a:{t:0 for (p,t) in mdp.T(s,a)}
                           for a in mdp.actlist}
                        for s in mdp.states},
               Nsa = {s:{a:0. for a in mdp.actlist}
                      for s in mdp.states},
               s = None,
               a = None)
        
    def program(self, percept):
        s1,r1 = percept
        mdp,U,s,a,Nsa,Ns_sa = self.mdp,self.U,self.s,self.a,self.Nsa,self.Ns_sa
        if s1 not in mdp.reward: # mdp.R also tracks the visited states
            U[s1] = r1
            mdp.reward[s1] = r1
        if s is not None:
            Nsa[s][a] += 1
            Ns_sa[s][a][s1] += 1
            for t in Ns_sa[s][a]:
                if Ns_sa[s][a][t] > 0:
                    self.mdp.T_set((s,a,t), Ns_sa[s][a][t] / Nsa[s][a])
        U = policy_evaluation(self.pi, U, mdp)
        if s1 in mdp.terminals:
            self.s, self.a = None, None
            return False
        else:
            self.s, self.a = s1, self.pi[s1]
            return self.a

class PassiveTDAgent(agents.Agent):
    """Passive (non-learning) agent that uses temporal differences to learn
    utility estimates. [Fig. 21.4]"""
    def __init__(self,mdp,pi,alpha=None):
        update(self,
               pi = pi,
               U = {s:0. for s in mdp.states},
               Ns = {s:0 for s in mdp.states},
               s = None,
               a = None,
               r = None,
               gamma = mdp.gamma,
               terminals = mdp.terminals)
        if alpha is None:
            alpha = lambda n: 60./(59+n) # page 837
        else:
            self.alpha = alpha
    def program(self,percept):
        s1,r1 = percept
        pi,U,Ns,s,a,r = self.pi,self.U,self.Ns,self.s,self.a,self.r
        alpha,gamma = self.alpha,self.gamma
        if s1 not in U: U[s1] = r1
        if s is not None:
            Ns[s] += 1
            U[s] += alpha(Ns[s])*(r+gamma*U[s1]-U[s])
        if s in self.terminals: self.s,self.a,self.r = None,None,None
        else: self.s,self.a,self.r = s1, pi[s1],r1
        return self.a

class QLearningAgent(agents.Agent):
    """Active TD agent that uses temporal differences to learn an
    action-utility representation. [Fig. 21.8]"""
    def __init__(self,mdp,alpha=None,Ne=5,Rplus=2):
        update(self,
               Q = {s:{a:0. for a in mdp.actlist}
                    for s in mdp.states if s not in mdp.terminals},
               Nsa = {s:{a:0. for a in mdp.actlist}
                    for s in mdp.states},
               s = None,
               a = None,
               r = None,
               Ne = Ne,
               Rplus = Rplus,
               gamma = mdp.gamma,
               terminals = mdp.terminals)

        for s in mdp.terminals: self.Q[s] = {None:0.}
        
        if alpha is None:
            self.alpha = lambda n: 60./(59+n) # page 837
        else:
            self.alpha = alpha
            
    def f(self,u,n): # the exploration function in AIMA(3rd ed), pg 842
        if n < self.Ne:
            return self.Rplus
        else:
            return u
            
    def program(self,percept):
        s1,r1 = percept
        Q, Nsa, s, a, r = self.Q, self.Nsa, self.s, self.a, self.r
        alpha, gamma, f = self.alpha, self.gamma, self.f
        if s1 in self.terminals:
            Q[s1][None] = r1
        if s is not None:
            Nsa[s][a] += 1
            Q[s][a] += alpha(Nsa[s][a])*(r+gamma*max(Q[s1].values())-Q[s][a])
        if s1 in self.terminals:
            self.s,self.a,self.r = None, None, None
            return False
        else:
            self.s,self.r = s1,r1
            self.a = argmax(Q[s1].keys(),lambda a1: f(Q[s1][a1],Nsa[s1][a1]))
            return self.a

# ---

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
        next_action = agent.program((current_state, current_reward))
        if next_action == False:
            break
        current_state = simulate(mdp,(current_state, next_action))

def demoPassiveADPAgent():
    print '--------------------'
    print 'DEMO PassiveADPAgent'
    print '--------------------'
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

    print '\nCorrect Utilities (estimated by value iteration, for comparison):'
    print value_iteration(Fig[17,1])

def demoPassiveTDAgent():
    print '--------------------'
    print 'DEMO PassiveTDAgent'
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

    print '\nCorrect Utilities (estimated by value iteration, for comparison):'
    print value_iteration(Fig[17,1])

def demoQLearningAgent():
    print '--------------------'
    print 'DEMO PassiveTDAgent'
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
    
    time_start = time()
    trials = 1000
    agent = QLearningAgent(Fig[17,1])
    for i in range (0,trials):
        execute_trial(agent,Fig[17,1])
    time_end = time()
    
    seconds_elapsed = time_end - time_start
    minutes_elapsed = seconds_elapsed / 60.0
    final_results = (('Took %d seconds, which is %d minutes' % (seconds_elapsed, minutes_elapsed)),\
                     ('Executed %i trials' % (trials)),
                     ('Utilities: %s' % {s:max(agent.Q[s].values()) for s in agent.Q}))
    for result in final_results:
        print result

    print '\nCorrect Utilities (estimated by value iteration, for comparison):'
    print value_iteration(Fig[17,1])

# ---

if __name__ == '__main__':
    demoPassiveADPAgent()
    demoPassiveTDAgent()
    demoQLearningAgent()

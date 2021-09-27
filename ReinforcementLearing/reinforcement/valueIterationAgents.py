# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        
        for iteration in range(self.iterations):
            #Temporaty dictionary for each iteration 
            Temp = util.Counter()
            for state in self.mdp.getStates():
                if(self.mdp.isTerminal(state) == False):

                    bestAction = self.computeActionFromValues(state)
                    Temp[state] = self.computeQValueFromValues(state, bestAction)

                else:
                    Temp[state] = self.mdp.getReward(state, 'exit', '')

            self.values = Temp

        
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        Sum = 0
        for nextState, Prob in self.mdp.getTransitionStatesAndProbs(state, action):
            Sum += Prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))

        return Sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestAction = 0
        bestValue = -999999
        for action in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, action)
            if(qValue > bestValue):
                bestValue = qValue
                bestAction = action
        
        return bestAction
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        Temp = 0

        for iteration in range(self.iterations):

            index = iteration % len(states)
            currentState = states[index]

            if(self.mdp.isTerminal(currentState) == False):
                bestAction = self.computeActionFromValues(currentState)
                Temp = self.computeQValueFromValues(currentState, bestAction)

                self.values[currentState] = Temp

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        
        predecessors = {}
        PQueue = util.PriorityQueue()   
        states = self.mdp.getStates()
        Temp = 0
        
        for state in states:
            if(self.mdp.isTerminal(state) == False):
                bestAction = self.computeActionFromValues(state)
                Temp = self.computeQValueFromValues(state, bestAction)

                diff = abs(Temp - self.getValue(state))
                PQueue.push(state, -1 * diff)

                for action in self.mdp.getPossibleActions(state):
                    for nextState, Prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in predecessors:
                            if state not in predecessors[nextState]:
                                predecessors[nextState].append(state)
                        else:
                            predecessors[nextState] = [state]

        for iteration in range(self.iterations):

            if(PQueue.isEmpty()):
                return 0
            currentState = PQueue.pop()
            if(self.mdp.isTerminal(currentState)):
                continue
            else:
                bestAction = self.computeActionFromValues(currentState)
                Temp = self.computeQValueFromValues(currentState, bestAction)
                self.values[currentState] = Temp

            for p in predecessors[currentState]:
                bestAction = self.computeActionFromValues(p)
                Temp = self.computeQValueFromValues(p, bestAction)

                diff = abs(Temp - self.getValue(p))
                if(diff > self.theta):
                    PQueue.update(p, -1 * diff)





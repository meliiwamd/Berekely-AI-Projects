# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        finalScore = 0
        nearestFood = 999999

        #action shouldn't be stop
        if action == 'Stop':
            return -999999
        #next state shouldn't be ghost
        for ghost in newGhostStates:
            if ghost.getPosition() == newPos:
                return -999999
        
        #now we consider the distance of each food
        #we want min
        for food in currentGameState.getFood().asList():
            manhattanDist = manhattanDistance(food, newPos)
            if manhattanDist < nearestFood:
                nearestFood = manhattanDist
        finalScore = -1 * nearestFood
        return finalScore
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the minimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxFinder(-1, 0, gameState)[0]

    def minFinder(self, reachedDepth, selectAgent ,gameState):
        if gameState.isLose() or gameState.isWin() or self.depth == reachedDepth:
            return None, self.evaluationFunction(gameState)
        else:
            best = (None, 9999)
            selectAgent = selectAgent + 1
            for Action in gameState.getLegalActions(selectAgent):
                nextState = gameState.generateSuccessor(selectAgent, Action)
                if selectAgent < gameState.getNumAgents() - 1:
                    Seleceted = (Action, self.minFinder(reachedDepth, selectAgent, nextState)[1])
                else:
                    Seleceted = (Action, self.maxFinder(reachedDepth, selectAgent, nextState)[1])
                if Seleceted[1] < best[1]:
                    best = Seleceted
            return best



    def maxFinder(self, reachedDepth, selectAgent ,gameState):
        selectAgent = 0
        reachedDepth = reachedDepth + 1
        if gameState.isLose() or gameState.isWin() or self.depth == reachedDepth:
            return None, self.evaluationFunction(gameState)
        else:
            
            best = (None, -9999)

            for Action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, Action)
                Seleceted = (Action, self.minFinder(reachedDepth, selectAgent, nextState)[1])
                if Seleceted[1] > best[1]:
                    best = Seleceted
            return best


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxFinder(-1, 0, gameState, -999999, 999999)[0]
        

    def minFinder(self, reachedDepth, selectAgent ,gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or self.depth == reachedDepth:
            return None, self.evaluationFunction(gameState)
        else:
            best = (None, 999999)
            selectAgent = selectAgent + 1
            for Action in gameState.getLegalActions(selectAgent):
                nextState = gameState.generateSuccessor(selectAgent, Action)
                if selectAgent < gameState.getNumAgents() - 1:
                    Seleceted = (Action, self.minFinder(reachedDepth, selectAgent, nextState, alpha, beta)[1])
                else:
                    Seleceted = (Action, self.maxFinder(reachedDepth, selectAgent, nextState, alpha, beta)[1])
                if Seleceted[1] < best[1]:
                    best = Seleceted
                beta = min(beta, best[1])
                if beta < alpha:
                    break
            return best



    def maxFinder(self, reachedDepth, selectAgent ,gameState, alpha, beta):
        selectAgent = 0
        reachedDepth = reachedDepth + 1
        if gameState.isLose() or gameState.isWin() or self.depth == reachedDepth:
            return None, self.evaluationFunction(gameState)
        else:
            
            best = (None, -999999)

            for Action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, Action)
                Seleceted = (Action, self.minFinder(reachedDepth, selectAgent, nextState, alpha, beta)[1])
                if Seleceted[1] > best[1]:
                    best = Seleceted
                alpha = max(alpha, best[1])
                if beta < alpha:
                    break
            return best

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best = (None, -9999)

        for Action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, Action)
            Seleceted = (Action, self.averageFinder(0, 0, nextState))
            if Seleceted[1] > best[1]:
                best = Seleceted
        return best[0]
        
    def averageFinder(self, reachedDepth, selectAgent ,gameState):
        if gameState.isLose() or gameState.isWin() or self.depth == reachedDepth:
            return self.evaluationFunction(gameState)
        else:
            average = 0
            selectAgent = selectAgent + 1
            for Action in gameState.getLegalActions(selectAgent):
                nextState = gameState.generateSuccessor(selectAgent, Action)
                if selectAgent < gameState.getNumAgents() - 1:
                    Seleceted = (Action, self.averageFinder(reachedDepth, selectAgent, nextState))
                else:
                    Seleceted = (Action, self.maxFinder(reachedDepth, selectAgent, nextState))
                average = average + Seleceted[1]
            return average/len(gameState.getLegalActions(selectAgent))

    "As It Was Before Should Choose The Max"

    def maxFinder(self, reachedDepth, selectAgent ,gameState):
        selectAgent = 0
        reachedDepth = reachedDepth + 1
        if gameState.isLose() or gameState.isWin() or self.depth == reachedDepth:
            return self.evaluationFunction(gameState)
        else:
            
            best = (None, -9999)

            for Action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, Action)
                Seleceted = (Action, self.averageFinder(reachedDepth, selectAgent, nextState))
                if Seleceted[1] > best[1]:
                    best = Seleceted
            return best[1]
        


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    finalScore = 0
    #we need 3 fors on each thing that may affect our direction and state
    foodVariable = 999999
    ghosts = 999999
    capsules = 999999

    #coefficients
    foodCoefficient = -5
    foodRemainingCoefficient = -60
    ghostCoefficient = 1
    capsuleRemainingCoefficient = -100
    capsuleCoefficient = -1


    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    ghostList = currentGameState.getGhostStates()
    currentPos = currentGameState.getPacmanPosition()
    remainingFood = foodRemainingCoefficient * len(foodList)
    remainingCapsule = capsuleRemainingCoefficient * len(capsuleList)

    #food
    for food in foodList:
        manhattanDist = manhattanDistance(food, currentPos)
        if manhattanDist < foodVariable:
            foodVariable = manhattanDist
    foodVariable = foodCoefficient * foodVariable
    if remainingFood == 0:
        foodVariable =  99999999

    #capsule
    for capsule in capsuleList:
        manhattanDist = manhattanDistance(capsule, currentPos)
        if manhattanDist < capsules:
            capsules = manhattanDist
    capsules = capsuleCoefficient * capsules

    if remainingCapsule == 0:   
       capsules = 0
    
    #ghosts
    for ghost in ghostList:
        manhattanDist = manhattanDistance(ghost.getPosition(), currentPos)
        if manhattanDist <= 1:
            return -999999
        if manhattanDist < ghosts:
            ghosts = manhattanDist
    ghosts = ghostCoefficient * ghosts
    finalScore = int(foodVariable) + int(ghosts) + int(capsules) + remainingCapsule + remainingFood

    return finalScore

# Abbreviation
better = betterEvaluationFunction

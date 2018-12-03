Minimum spanning trees
    - Is a subset of the edges of a connected, edge-weighted undirected graph that connects all the 
    vertices together, without any cycles and with the minimum possible total edge weight

    Greedy algorithm
        - Finds the minimum spanning tree by always choosing the minimum weight to connect vertices 

        Kruskal algorithm
            - Finds a MST by choosing the minimum weight edge in a graph as long as: there are edges to
            choose from, the edge doesnt form a cycle, and we still haven't found a MST

        Prim's algorithm
            - Finds a MST by randomly choosing a vertex to begin building the MST, and then iteratively add
            the minimum edge that connects a vertex not yet in the tree to a vertex in the tree


Dijkstras algorithm
    - Used to find the shortest paths between nodes in a graph. Start with an initial node
    and then visit every node updating the distance from the initial node to that node every time
    a shorter path is found.
    
Search algorithms
    - Algorithms for traversing or searching a tree or a graph

    Depth first search
        - Starts at the root node and it explores as far as possible along each branch before backtracking.  

    Breadth first search
        - Starts at the root node and it explores all the neighbor nodes to the current node before 
        moving to the next depth level.

    Uniform cost search
        - Similar to Dijkstras algorithm, but with the difference that we do have a goal state, and not all the 
        nodes in the graph need to be explored, just the ones that lead to the goal. The algorithm starts at the root node,
        and it explores nodes in order of the minimum cost back to the root. In other words, it explores the minimum cost 
        vertices to the root vertex. UCS is complete and optimal, but it explores options in every “direction” and has no
        information about the goal location.

Infromed search

    Heuristic
        - Function for how close to goal we are.

    Greedy search
        - Expand the node that seems closer using a heuristic function. The problem is that it can first lead you to the
        wrong goal, and on the worst case scenario, it would ressemble a badly guided DFS, where all nodes have been explored 
        besides the path to the goal. 

    A* search
        - It can be thought as a combination of UCS and Greedy. UCS orders by path cost (backward cost), while Greedy orders
        by goal proximity (forward cost). A* search orders by the sum of these two costs. In tree seach, A* is optimal if heuristic
        is admissible. In Graph search, A* is optimal if heuristic is consistent. Often uses relaxed problems.

        Relaxed problems
            - Approximation of a difficult problem that is easier to solve. Once solved, the solution provides information about the
            original problem.

        Admissible heuristic
            - It never overestimates the cost of reaching a goal, i.e. the cost estimate to the goal is not higher than the minimum
            cost from the current point to the goal.

        Consistent Heuristic
            - Estimate cost is always less than or equal to the estimated distance from any neighboring vertex to the goal, plus the cost 
            of reaching that neighbor.

Adverserial search

    Minimax
        - Decision rule used for minimizing the possible loss for a worst case scenario. One player maximizes the result, while the other
        minimizes the result. It basically consists on a state-space search tree where agents alternate turns (depth level per turn) and 
        at each node, the "minimax" value is computed. The minimax value is the best achievable utility against a rational adversary.

        Depth limited search and evaluation function
            - In most realistic games, we cannot search a minimax tree to the leaves level. Therefore, a depth limiting value is used in
            conjunction with an evaluation function that estimates a score on a non-terminal state. 

        Alpha Beta pruning 
            - Seeks to decrease the number of nodes that are evaluated by the minimax algorithm in its search tree. It basically avoids expanding nodes
            that wont be played because they will be ignored by the min/max agent(s). 

Uncertainty and utilities
    - Uncertain outcomes controlled by chance, instead of an adversary.

    Expectimax search
        - Similiar to Minimax, but values should now reflect average-case (expectimax) outcomes, not worst-case (minimax) outcomes. On a expectimax search, we aim to compute the average score under optimal play. The max nodes remain the same as in minimax, but the min nodes are now
        "chance" nodes with a calculated weighted average of its children. This weighted average is tied to the probability of the children node happening. In expectimax, we have a probabilistic model of how opponents will behave at a given state. 

Markov decision processes    
    - Defined by a set of states, a set of actions, a transition function, a reward function, a start state and a terminal state. MDPs are 
    non-deterministic search problems. Action outcomes depend only on the current state, i.e. at the present state, the future and past are independent.
    Unlike deterministic problems where we want an optimal plan from start to goal, for MDPs, we want an optimal policy. 
    
    Policy 
        - Action for each state. 

    Utility
        - Sum of (discounted) rewards.

    Value
        - Expected future utility from a state (max node).

    Q-Values
        - Expected future utility from a q-state (chance node). 

    Discounting
        - Values of rewards decay exponentially at each successor state. We do this because sooner rewards probably do have higher utility than later rewards and it also helps our algorithms converge

    Time limited values
        - When we a expectimax tree goes on forever, we can do a depth-limited computation with increasing depths until change is small. Deep parts of the tree eventually don’t matter if the discount is less than 1. Define Vk(s) to be the optimal value of s if the game ends in k more time steps.

    Value iteration
        - Start with V0(s) = 0: no time steps left means an expected reward sum of zero. Given vector of Vk(s) values, do one ply of expectimax from each state. Repeat until convergence.

    Bellman Equations
        - Used to characterize optimal values. 

    Policy evalutaion
        - Turn recursive Bellman equations into updates(like value iteration)

    Policy Extraction
        - Gets the policy implied by the values

    Policy Iteration
        - Its still optimal and it can converge (much) faster under some conditions. It consists of: 
            1 - Policy evaluation: calculate utilities for some fixed policy (not optimal utilities!) until convergence
            2 - Policy improvement: update policy using one-step look-ahead with resulting converged (but not optimal!) utilities as future values
            3 - Repeat steps until policy converges
    
    MDPs Summary
        1 - Compute optimal values: use value iteration or policy iteration
        2 - Compute values for a particular policy: use policy evaluation
        3 - Turn your values into a policy: use policy extraction (one-step lookahead)

    Offline planning
        - Solving MDPs is offline planning because you determine quantities through computation, you need to know the details of the MDP and you don't 
        actually play the game.

Reinforcement learning
    - Receive feedback in the form of rewards. Agent's utility is defined by the reward function. Must (learn to) act so as to maximize expected rewards. All learning is based on observed samples of outcomes. It's basically an MDP without a transition model nor a reward function, i.e. we must actually try actions to learn what actions really do and what the state's rewards are. 

    Model-based learning
        - Learn an approximate model based on experiences. Solve for values as if the learned model were correct. In other words, first learn a model by learining from the outcomes for each state reached and action taken. Normalize this data to provide estimates for transition, and record the reward data at each state. Once a model is estimated, solve the learned MDP with value or policy iteration.

    Model-free learning
        
        Passive RL
            - Start with a fixed policy and try to learn the state values. The learner executes the fixed policy and learn from experience. Unlike offline planning, you actually take actions in the world. To compute the values for each state, we use Direct Evaluation.

            Direct evaluation
                - Act according to your fixed policy. Every time you visit a state, write down what the sum of discounted rewards and average those samples. The problem with this is that it takes a long time to learn because it wastes information about state connections and each state must be learned separately

            Temporal difference learning
                - Try to learn from every experience. Update V(s) each time we experience a transition (s, a, s’, r). Likely outcomes s’ will contribute updates more often. Policy still fixed, still doing evaluation. Move values toward value of whatever successor occurs: running average. By using an exponential running average, we make recent samples more important and eventually forgets the past (wrong old values).

        Active RL
            - You don’t know the transitions T(s,a,s’) nor the rewards R(s,a,s’). You choose which actions to take each time with the goal of learning the optimal policy / values. A big issue with active RL is exploration vs. exploitation.

            Q-Learning (off-policy learning)
                - Learn Q(s,a) values as you go instead of the actual value. Receive a sample (s,a,s’,r), consider your old estimate and incorporate the new estimate into a running average. Q-learning converges to optimal policy even if you’re acting suboptimally.

            Exploration
                - Act reandomly to explore other states

            Exploitation
                - Act according to your current policy

            Regret
                - Total mistake cost

            Feature based representation
                - Write the q-function (or value function) for any state using a wieghted sum. Advantage: our experience is summed up in a few powerful numbers. Disadvantage: states may share features but actually be very different in value.


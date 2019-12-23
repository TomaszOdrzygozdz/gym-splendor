"""Monte Carlo Tree Search for stochastic environments."""

import asyncio
import random

import gin
import gym

from alpacka import data
from alpacka.agents import base
from alpacka.agents import core


@gin.configurable
def rate_new_leaves_with_rollouts(
    leaf,
    observation,
    model,
    discount,
    rollout_agent_class=core.RandomAgent,
    rollout_time_limit=100,
):
    """Basic rate_new_leaves_fn based on rollouts with an Agent.

    Args:
        leaf (TreeNode): Node whose children are to be rated.
        observation (np.ndarray): Observation received at leaf.
        model (gym.Env): Model environment.
        discount (float): Discount factor.
        rollout_agent_class (type): Agent class to use for rollouts.
        rollout_time_limit (int): Maximum number of timesteps for rollouts.

    Yields:
        Network prediction requests.

    Returns:
        list: List of pairs (reward, value) for all actions played from leaf.
    """
    del leaf
    agent = rollout_agent_class(model.action_space)
    init_state = model.clone_state()

    child_ratings = []
    for init_action in range(model.action_space.n):
        (observation, init_reward, done, _) = model.step(init_action)
        value = 0
        total_discount = 1
        time = 0
        while not done and time < rollout_time_limit:
            (action, _) = yield from agent.act(observation)
            (observation, reward, done, _) = model.step(action)
            value += total_discount * reward
            total_discount *= discount
            time += 1
        child_ratings.append((init_reward, value))
        model.restore_state(init_state)
    return child_ratings


@gin.configurable
def rate_new_leaves_with_value_network(leaf, observation, model, discount):
    """rate_new_leaves_fn based on a value network (observation -> value)."""
    del leaf
    del observation

    init_state = model.clone_state()

    def step_and_rewind(action):
        (observation, reward, done, _) = model.step(action)
        model.restore_state(init_state)
        return (observation, reward, done)

    (observations, rewards, dones) = data.nested_stack([
        step_and_rewind(action) for action in range(model.action_space.n)
    ])
    # Run the network to predict values for children.
    values = yield observations
    # Compute the final ratings, masking out "done" states.
    return rewards + discount * values * (1 - dones)


class TreeNode:
    """Node of the search tree.

    Attrs:
        children (list): List of children, indexed by action.
        is_leaf (bool): Whether the node is a leaf, i.e. has not been expanded
            yet.
        is_terminal (bool): Whether the node is terminal, i.e. the environment
            returns "done" when stepping into this state. For now we assume that
            "done"s are deterministic.
            TODO(koz4k): Lift this assumption.
        graph_node (GraphNode): The corresponding graph node - many to one
            relation.
    """

    def __init__(self, init_reward, init_value=None):
        """Initializes TreeNode.

        Args:
            init_reward (float): Reward collected when stepping into the node
                the first time.
            init_value (float or None): Value received from a rate_new_leaves_fn
                for this node, or None if it's the root.
        """
        self._reward_sum = init_reward
        self._reward_count = 1
        self._init_value = init_value
        self._graph_node = None
        self.children = None
        self.is_terminal = False

    def init_graph_node(self, graph_node=None):
        """Assigns the node's GraphNode, or creates a new one."""
        assert self._graph_node is None, 'Graph node initialized twice.'
        if graph_node is None:
            graph_node = GraphNode(self._init_value)
        self._graph_node = graph_node

    @property
    def graph_node(self):
        return self._graph_node

    def visit(self, reward, value):
        """Records a visit in the node during backpropagation.

        Args:
            reward (float): Reward collected when stepping into the node.
            value (float or None): Value accumulated on the path out of the
                node, or None if value should not be accumulated.
        """
        self._reward_sum += reward
        self._reward_count += 1
        # Terminal nodes don't have GraphNodes assigned, so don't update value.
        if not self.is_terminal and value is not None:
            assert self.graph_node is not None, (
                'Graph node must be assigned first.'
            )
            self.graph_node.visit(value)

    def quality(self, discount):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        Quality(s, a) = reward(s, a) + discount * value(s').
        """
        return self._reward_sum / self._reward_count + discount * (
            self._graph_node.value
            if self._graph_node is not None else self._init_value
        )

    @property
    def is_leaf(self):
        return self.children is None


class GraphNode:
    """Node of the search graph.

    In the graph mode, corresponds to a state in the MDP. Outside of the graph
    mode, corresponds 1-1 to a TreeNode.

    Attrs:
        value (float): Value accumulated in this node.
    """

    def __init__(self, init_value):
        """Initializes GraphNode.

        Args:
            init_value (float or None): Value received from a rate_new_leaves_fn
                for this node, or None if it's the root.
        """
        self._value_sum = 0
        self._value_count = 0
        if init_value is not None:
            self.visit(init_value)
        # TODO(koz4k): Move children here?

    def visit(self, value):
        """Records a visit in the node during backpropagation.

        Args:
            value (float): Value accumulated on the path out of the node.
        """
        self._value_sum += value
        self._value_count += 1

    @property
    def value(self):
        return self._value_sum / self._value_count


class DeadEnd(Exception):
    """Exception raised in case of a dead end.

    Dead end occurs when every action leads to a loop.
    """


class StochasticMCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search for stochastic environments.

    For now it also supports transpositions and loop avoidance for
    deterministic environments.
    TODO(koz4k): Merge those features with DeterministicMCTSAgent. Add features
    specific to stochastic environments to StochasticMCTSAgent.
    """

    def __init__(
        self,
        action_space,
        n_passes=10,
        discount=0.99,
        rate_new_leaves_fn=rate_new_leaves_with_rollouts,
        graph_mode=False,
        avoid_loops=False,
        loop_penalty=0,
    ):
        """Initializes MCTSAgent.

        Args:
            action_space (gym.Space): Action space.
            n_passes (int): Number of MCTS passes per act().
            discount (float): Discount factor.
            rate_new_leaves_fn (callable): Coroutine estimating rewards and
                values of new leaves. Can ask for predictions using a Network.
                Should return rewards and values for every child of a given leaf
                node. Signature:
                (leaf, observation, model, discount) -> [(reward, value)].
            graph_mode (bool): Turns on using transposition tables, turning the
                search graph from a tree to a DAG.
            avoid_loops (bool): Prevents going back to states already visited on
                the path from the root.
            loop_penalty (float): Value backpropagated from "dead ends" - nodes
                from which it's impossible to reach a node that hasn't already
                been visited.
        """
        assert isinstance(action_space, gym.spaces.Discrete), (
            'MCTSAgent only works with Discrete action spaces.'
        )
        super().__init__(action_space)

        if avoid_loops:
            assert graph_mode, 'Loop avoidance only works in graph mode.'

        self.n_passes = n_passes
        self._discount = discount
        self._rate_new_leaves = rate_new_leaves_fn
        self._graph_mode = graph_mode
        self._avoid_loops = avoid_loops
        self._loop_penalty = loop_penalty
        self._model = None
        self._root = None
        self._root_state = None
        self._real_visited = None
        self._state_to_graph_node = {}

    def _rate_children(self, node):
        """Returns qualities of all children of a given node."""
        return [child.quality(self._discount) for child in node.children]

    def _choose_action(self, node, visited):
        """Chooses the action to take in a given node based on child qualities.

        If avoid_loops is turned on, tries to avoid nodes visited on the path
        from the root.

        Args:
            node (TreeNode): Node to choose an action from.
            visited (set): Set of GraphNodes visited on the path from the root.

        Returns:
            Action to take.

        Raises:
            DeadEnd: If there's no child not visited before.
        """
        # TODO(koz4k): Distinguish exploratory/not.
        child_qualities = self._rate_children(node)
        child_qualities_and_actions = zip(
            child_qualities, range(len(child_qualities))
        )

        if self._avoid_loops:
            # Filter out nodes visited on the path from the root.
            child_graph_nodes = [child.graph_node for child in node.children]
            child_qualities_and_actions = [
                (quality, action)
                for (quality, action) in child_qualities_and_actions
                if child_graph_nodes[action] not in visited
            ]

        if not child_qualities_and_actions:
            # No unvisited child - dead end.
            raise DeadEnd

        (_, action) = max(child_qualities_and_actions)
        return action

    def _traverse(self, root, observation, path):
        """Chooses a path from the root to a leaf in the search tree.

        Does not modify the nodes.

        Args:
            root (TreeNode): Root of the search tree.
            observation (np.ndarray): Observation received at root.
            path (list): Empty list that will be filled with pairs
                (reward, node) of nodes visited during traversal and rewards
                collected when stepping into them. It is passed as an argument
                rather than returned, so we can access the result in case of
                a DeadEnd exception.

        Returns:
            Tuple (observation, done, visited), where observation is the
            observation received in the leaf, done is the "done" flag received
            when stepping into the leaf and visited is a set of GraphNodes
            visited on the path. In case of a "done", traversal is interrupted.
        """
        assert not path, 'Path accumulator should initially be empty.'
        path.append((0, root))
        visited = {root.graph_node}
        node = root
        done = False
        visited = set()
        while not node.is_leaf and not done:
            action = self._choose_action(node, visited)
            node = node.children[action]
            (observation, reward, done, _) = self._model.step(action)
            path.append((reward, node))
            visited.add(node.graph_node)
        return (observation, done, visited)

    def _expand_leaf(self, leaf, observation, done, visited):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial rewards and values. The
        reward and value of the "best" new leaf is then backpropagated.

        Only modifies leaf - assigns a GraphNode and adds children.

        Args:
            leaf (TreeNode): Leaf to expand.
            observation (np.ndarray): Observation received at leaf.
            done (bool): "Done" flag received at leaf.
            visited (set): Set of GraphNodes visited on the path from the root.

        Yields:
            Network prediction requests.

        Returns:
            float: Quality of a chosen child of the expanded leaf, or None if we
            shouldn't backpropagate quality beause the node has already been
            visited.
        """
        assert leaf.is_leaf

        if done:
            leaf.is_terminal = True
            # In a "done" state, cumulative future return is 0.
            return 0

        already_in_graph = False
        if self._graph_mode:
            state = self._model.clone_state()
            graph_node = self._state_to_graph_node.get(state, None)
            leaf.init_graph_node(graph_node)
            if graph_node is not None:
                already_in_graph = True
            else:
                self._state_to_graph_node[state] = leaf.graph_node
        else:
            leaf.init_graph_node()

        child_ratings = yield from self._rate_new_leaves(
            leaf, observation, self._model, self._discount
        )
        leaf.children = [
            TreeNode(reward, value) for (reward, value) in child_ratings
        ]
        if already_in_graph:
            # Node is already in graph, don't backpropagate quality.
            return None
        action = self._choose_action(leaf, visited)
        return leaf.children[action].quality(self._discount)

    def _backpropagate(self, value, path):
        """Backpropagates value to the root through path.

        Only modifies the rewards and values of nodes on the path.

        Args:
            value (float or None): Value collected at the leaf, or None if value
                should not be backpropagated.
            path (list): List of (reward, node) pairs, describing a path from
                the root to a leaf.
        """
        for (reward, node) in reversed(path):
            node.visit(reward, value)
            if value is not None:
                value = reward + self._discount * value

    def _run_pass(self, root, observation):
        """Runs a pass of MCTS.

        A pass consists of:
            1. Tree traversal to find a leaf.
            2. Expansion of the leaf, adding its successor states to the tree
               and rating them.
            3. Backpropagation of the value of the best child of the old leaf.

        During leaf expansion, new children are rated only using
        the rate_new_leaves_fn - no actual stepping into those states in the
        environment takes place for efficiency, so that rate_new_leaves_fn can
        be implemented by running a neural network that rates all children of
        a given node at the same time. This means that those new leaves cannot
        be matched with GraphNodes yet, which leads to the special case 2.
        below.

        Special cases:
            1. In case of a "done", traversal is interrupted, the leaf is not
               expanded and value 0 is backpropagated.
            2. When graph_mode is turned on and the expanded leaf turns out to
               be an already visited node in the graph, we don't backpropagate
               value (because it's unclear which one to backpropagate), so the
               next pass can follow the same path, go through this node and find
               a "true" leaf. Rewards are backpropagated however, because
               they're still valid.
            3. When avoid_loops is turned on and during tree traversal no action
               can be chosen without making a loop, value defined in the
               loop_penalty  option is backpropagated from the node.

        Args:
            root (TreeNode): Root node.
            observation (np.ndarray): Observation collected at the root.

        Yields:
            Network prediction requests.
        """
        path = []
        try:
            (observation, done, visited) = self._traverse(
                root, observation, path
            )
            (_, leaf) = path[-1]
            quality = yield from self._expand_leaf(
                leaf, observation, done, visited
            )
        except DeadEnd:
            quality = self._loop_penalty
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

    @asyncio.coroutine
    def reset(self, env, observation):
        """Reinitializes the search tree for a new environment."""
        del observation
        assert env.action_space == self._action_space
        self._model = env
        # Initialize root with some reward to avoid division by zero.
        self._root = TreeNode(init_reward=0)
        self._real_visited = set()

    def act(self, observation):
        """Runs n_passes MCTS passes and chooses the best action."""
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self.n_passes):
            yield from self._run_pass(self._root, observation)

        # Add the root to visited nodes after running the MCTS passes to ensure
        # it has a graph node assigned.
        self._real_visited.add(self._root.graph_node)

        try:
            # Avoid the nodes already visited on the path in the real
            # environment when choosing an action.
            action = self._choose_action(self._root, self._real_visited)
        except DeadEnd:
            action = random.randrange(len(self._root.children))
        self._root = self._root.children[action]
        return (action, {})

"""Agent base classes."""

import asyncio

from alpaca.alpacka import data
from alpaca.alpacka import envs


class Agent:
    """Agent base class.

    Agents can use neural networks internally. Network prediction is run outside
    of the Agent, so it can be batched across multiple Agents for efficiency.
    This is done using a coroutine API, explained in solve().
    """

    def __init__(self):
        """Initializes Agent.

        Args:
            action_space (gym.Space): Action space. It's passed in the
                constructor instead of being inferred from env in solve(),
                because it shouldn't change between environments and this way
                the API for stateless OnlineAgents is simpler.
        """

    def solve(self, env, init_state=None, time_limit=None):
        """Solves a given environment.

        Coroutine, suspends execution for every neural network prediction
        request. This enables a very convenient interface for requesting
        predictions by the Agent:

            def solve(self, env, init_state=None):
                # Planning...
                predictions = yield inputs
                # Planning...
                predictions = yield inputs
                # Planning...
                return episode

        Example usage:

            coroutine = agent.solve(env)
            try:
                prediction_request = next(coroutine)
                network_output = process_request(prediction_request)
                prediction_request = coroutine.send(network_output)
                # Possibly more prediction requests...
            except StopIteration as e:
                episode = e.value

        Agents that do not use neural networks should wrap their solve() method
        in an @asyncio.coroutine decorator, so Python knows to treat it as
        a coroutine even though it doesn't have any yield.

        Args:
            env (gym.Env): Environment to solve.
            init_state (object): Reset the environment to this state.
                If None, then do normal gym.Env.reset().
            time_limit (int or None): Maximum number of steps to make on the
                solved environment. None means no time limit.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            (Agent/Trainer-specific) Episode object summarizing the collected
            data for training the Network.
        """
        raise NotImplementedError


class OnlineAgent(Agent):
    """Base class for online agents, i.e. planning on a per-action basis.

    Provides a default implementation of Agent.solve(), returning a Transition
    object with the collected batch of transitions.
    """

    @asyncio.coroutine
    def reset(self, env, observation):  # pylint: disable=missing-param-doc
        """Resets the agent state.

        Called for every new environment to be solved. Overriding is optional.

        Args:
            env (gym.Env): Environment to solve.
            observation (Env-dependent): Initial observation returned by
                env.reset().
        """

    def act(self, observation):

        """Determines the next action to be performed.

        Coroutine, suspends execution similarly to Agent.solve().

        In model-based agents, the original environment state MUST be restored
        in the end of act(). This is not checked at runtime, since it would be
        a big overhead for heavier environments.

        Args:
            observation (Env-dependent): Observation from the environment.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Pair (action, agent_info), where action is the action to make in the
            environment and agent_info is a dict of additional info to be put as
            Transition.agent_info.
        """
        raise NotImplementedError

    @staticmethod
    def postprocess_transition(transition):
        """Postprocesses Transitions before passing them to Trainer.

        Can be overridden in subclasses to customize data collection.

        Called after the episode has finished, so can incorporate any
        information known only in the hindsight to the transitions.

        Args:
            transition (Transition): Transition to postprocess.

        Returns:
            Postprocessed Transition.
        """
        return transition

    def solve(self, env, init_state=None, time_limit=None):
        """Solves a given environment using OnlineAgent.act().

        Args:
            env (gym.Env): Environment to solve.
            init_state (object): Reset the environment to this state.
                If None, then do normal gym.Env.reset().
            time_limit (int or None): Maximum number of steps to make on the
                solved environment. None means no time limit.

        Yields:
            Network-dependent: A stream of Network inputs requested for
            inference.

        Returns:
            data.Episode: Episode object containing a batch of collected
            transitions and the return for the episode.
        """
        model_env = env

        if time_limit is not None:
            # Add the TimeLimitWrapper _after_ passing the model env to the
            # agent, so the states cloned/restored by the agent do not contain
            # the number of steps made so far - this would break state lookup
            # in some Agents.
            env = envs.TimeLimitWrapper(env, time_limit)

        if init_state is None:
            # Model-free case...
            observation = env.reset()
        else:
            # Model-based case...
            observation = env.restore_state(init_state)

        yield from self.reset(model_env, observation)

        transitions = []
        done = False
        info = {}
        while not done:
            # Forward network prediction requests to BatchStepper.
            (action, agent_info) = yield from self.act(observation)
            (next_observation, reward, done, info) = env.step(action)

            transitions.append(data.Transition(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
                agent_info=agent_info,
            ))
            observation = next_observation

        transitions = [
            self.postprocess_transition(transition)
            for transition in transitions
        ]

        return_ = sum(transition.reward for transition in transitions)
        solved = info['solved'] if 'solved' in info else None
        transition_batch = data.nested_stack(transitions)
        return data.Episode(
            transition_batch=transition_batch,
            return_=return_,
            solved=solved,
        )

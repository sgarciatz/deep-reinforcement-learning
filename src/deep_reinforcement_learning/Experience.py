import torch


class Experience(object):


    """A state, action, reward, new_state and priority tuple.

    An Experience is a data structure that holds the (s,a,r,s',p) tuple
    (state, action, reward, new_state, priority).

    Attributes:
        state: is the current state of the env.
        action: the action taken in the current env.
        reward: the reward associated to taking said action in the
            current environment.
        next_state: the resulting state after applying the action.
        done: a flag that states if a state is terminal.
        priority: The value of the temporal difference error for that
            experience. It is used for Prioritized Experience Replay.
    """

    def __init__(self,
                 state,
                 action,
                 reward,
                 next_state,
                 done,
                 priority: float = 1.0e2):
        """Create a new Experience, i.e. a (s,a,r,s',d,p) tuple."""
        self._state: torch.Tensor | tuple = state
        self._action: torch.Tensor = action
        self._reward: torch.Tensor = reward
        self._next_state: torch.Tensor = next_state
        self._done: torch.Tensor = done
        self._priority: torch.Tensor = torch.scalar_tensor(priority)

    @property
    def state(self) -> torch.Tensor | tuple:
        """Returns the state of the experience as a tensor or as a tuple
        of tensor if any variation of a QGraphNetwork is used.

        Returns:
            torch.Tensor | tuple: a tensor or as a tuple of tensors if
                any variation of a QGraphNetwork is used.
        """

        if (isinstance(self._state, tuple)):
            return (self._state[0],
                    self._state[1],
                    self._state[2])
        else:
            return self._state

    @property
    def action(self) -> torch.Tensor:
        """Return a tensor containing the action taken.

        Returns:
            torch.Tensor: A tensor containing the action taken.
        """
        return self._action

    @property
    def reward(self) -> torch.Tensor:
        """Return a tensor containing the reward given.

        Returns:
            torch.Tensor: A tensor containing the reward given.
        """
        return self._reward

    @property
    def next_state(self) -> torch.Tensor | tuple:
        """Returns the next_state of the experience as a tensor or as a
        tuple of tensor if any variation of a QGraphNetwork is used.

        Returns:
            torch.Tensor | tuple: a tensor or as a tuple of tensors if
                any variation of a QGraphNetwork is used.
        """

        if (isinstance(self._state, tuple)):
            return (self._next_state[0],
                    self._next_state[1],
                    self._next_state[2])
        else:
            return self._next_state

    @property
    def done(self) -> torch.Tensor:
        """Return a tensor containing wether the state is terminal or
        not.

        Returns:
            torch.Tensor: Wether the state is terminal or not.
        """
        return self._done

    @property
    def priority(self) -> torch.Tensor:
        """Returns a tensor containing the priority of the experience.

        Returns:
            torch.Tensor: a tensor containing the priority of the
                experience.
        """
        return self._priority

    @priority.setter
    def priority(self, new_priority: torch.Tensor):
        """Updates the priority of the experience

        Args:
            new_priority (int | torch.Tensor): The new priority of the
            experience.
        """
        self._priority = new_priority

    def __str__(self) -> str:
        """Parses the object information into a human-readable string.
        """
        string = "Experience:\n"
        string += f"\tState: {self.state}\n"
        string += f"\tAction: {self.action}\n"
        string += f"\tReward: {self.reward}\n"
        string += f"\tNext state: {self.next_state}\n"
        string += f"\tIs terminal: {self.done}\n"
        string += f"\tPriority: {self.priority}\n"
        return string

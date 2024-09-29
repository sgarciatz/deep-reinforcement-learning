from collections import deque
import random
import torch
from deep_reinforcement_learning.Experience import Experience
from abc import ABC, abstractmethod


class ExperienceMemory(ABC):


    """
    This class is the responsible of providing the experience batches
    to the DQLearning agent.

    It samples the experiences from the experiences buffer according to
    a sampling policy.

    Experiences are tuples of (s,a,r,s',p) and the buffer is
    implemented as a queue.
    """

    def __init__(self,
                 buffer_max_size: int = 10240,
                 device: str = "cpu") -> None:

        """

        """

        self.experience_buffer: deque[Experience] =\
            deque([], maxlen=buffer_max_size)
        self.device: str = device


    def add_experience(self, experience: Experience) -> None:
        """Inserts an experience into ``experience_buffer``.

        Args:
            experience (Experience): The experience to insert into
                ``experience_buffer``.
        """
        self.experience_buffer.append(experience)

    @abstractmethod
    def sample_experience(self, n: int = 1) -> list:
        ...

    @abstractmethod
    def update_batch_priorities(self,
                                batch: list,
                                td_error:torch.Tensor) -> None:
        ...

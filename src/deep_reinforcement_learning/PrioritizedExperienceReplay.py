from deep_reinforcement_learning.Experience import Experience
from deep_reinforcement_learning.ExperienceMemory import ExperienceMemory
import torch
import numpy as np
from random import choices
from collections import deque


class PrioritizedExperienceReplay(ExperienceMemory):

    def __init__(self,
                 buffer_max_size: int = 10240,
                 device: str = "cpu",
                 epsilon: float = 1e-4,
                 alpha: float = 1
                 ) -> None:
        """_summary_

        Args:
            buffer_max_size (int, optional): _description_. Defaults to 10240.
            device (str, optional): _description_. Defaults to "cpu".
            epsilon (float, optional): _description_. Defaults to 1e-4.
            alpha (float, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        super().__init__(buffer_max_size=buffer_max_size,
                         device=device)
        self.epsilon: torch.Tensor = torch.tensor(epsilon).to(self.device)
        self.alpha: torch.Tensor = torch.tensor(alpha).to(self.device)
        self._priorities: deque[float] = deque([], maxlen=buffer_max_size)


    def add_experience(self, experience: Experience):
        super().add_experience(experience)
        self._priorities.append(experience.priority)

    def sample_experience(self, n_samples: int = 1) -> list[Experience]:
        """Samples ``n_samples`` from the ``experience_buffer``
        according to their priority.

        Args:
            n_samples (int, optional): The number of samples to
                return. Defaults to 1.

        Returns:
            list[Experience]: The samples.
        """
        samples: list[Experience] = choices(self.experience_buffer,
                                            k = n_samples,
                                            weights= self._priorities)

        return samples

    def update_batch_priorities(self,
                                batch: list[Experience],
                                td_error: torch.Tensor) -> None:
        """Implementation of Prioritized Replay Memory.

        Args:
            batch (_type_): _description_
            td_error (_type_): _description_
        """
        priorities: torch.Tensor = torch.tensor(
            self._priorities,
            dtype=torch.float32).to(device=self.device)
        denominator: torch.Tensor = priorities + self.epsilon
        denominator = pow(denominator, self.alpha)
        denominator = sum(denominator)
        numerator: torch.Tensor = td_error + self.epsilon
        numerator = pow(numerator, self.alpha)
        new_priorities: torch.Tensor = numerator / denominator
        for i, new_p in enumerate(new_priorities):
            batch[i].priority = new_p.item()

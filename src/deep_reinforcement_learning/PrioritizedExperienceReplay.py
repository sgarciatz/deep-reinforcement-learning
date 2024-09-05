from deep_reinforcement_learning.Experience import Experience
from deep_reinforcement_learning.ExperienceMemory import ExperienceMemory
import torch
import numpy as np
from random import choices

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
        self.epsilon = torch.tensor(epsilon).to(self.device)
        self.alpha = torch.tensor(alpha).to(self.device)
        self._priorities = torch.tensor(
            [e.priority for e in self.experience_buffer],
            dtype=torch.float32)


    def add_experience(self, experience: Experience):
        super().add_experience(experience)
        self._priorities = torch.tensor(
            [e.priority for e in self.experience_buffer],
            dtype=torch.float32)
        return

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


        states = torch.tensor([e.state for e in samples],
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in samples],
                               dtype=torch.int64).to(self.device)

        next_states = torch.tensor([e.next_state for e in samples],
                                   dtype=torch.float32).to(self.device)
        rewards = torch.tensor([e.reward for e in samples],
                                dtype=torch.float32).to(self.device)
        dones = torch.tensor([e.done for e in samples],
                                dtype=torch.float32).to(self.device)
        priorities = torch.tensor([e.priority for e in samples],
                                dtype=torch.float32).to(self.device)

        return {"states": states,
                "actions": actions,
                "next_states": next_states,
                "rewards": rewards,
                "dones": dones,
                "priorities": priorities}

    def update_batch_priorities(self,
                                batch: list[Experience],
                                td_error: torch.Tensor) -> None:
        """Implementation of Prioritized Replay Memory.

        Args:
            batch (_type_): _description_
            td_error (_type_): _description_
        """
        denominator = self._priorities + self.epsilon
        denominator = pow(denominator, self.alpha)
        denominator = sum(denominator)
        numerator = td_error + self.epsilon
        numerator = pow(numerator, self.alpha)
        new_priorities = numerator / denominator
        for i, new_p in enumerate(new_priorities):
            batch[i].priority = new_p.item()
        self._priorities = torch.tensor(
            [e.priority for e in self.experience_buffer],
            dtype=torch.float32)

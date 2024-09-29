from deep_reinforcement_learning.Experience import Experience
from deep_reinforcement_learning.ExperienceMemory import ExperienceMemory
from torch import Tensor, tensor, float32, int32
import numpy as np
from random import sample

class UniformExperienceReplay(ExperienceMemory):

    def __init__(self,
                 buffer_max_size: int = 10240,
                 device: str = "cpu") -> None:
        """_summary_

        Args:
            buffer_max_size (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".
            epsilon (float, optional): _description_. Defaults to 1e-4.
            alpha (float, optional): _description_. Defaults to 1.
        """
        super().__init__(buffer_max_size=buffer_max_size,
                         device=device)

    def sample_experience(self, n_samples: int = 1) -> list[Experience]:
        """Samples ``n_samples`` from the ``experience_buffer``.

        Args:
            n_samples (int, optional): The number of samples to
                return. Defaults to 1.

        Returns:
            list[Experience]: The samples.
        """
        samples: list[Experience] = sample(self.experience_buffer,
                                            n_samples)
        return samples

    def update_batch_priorities(self, *args) -> None:
        """Do nothing since in ``UniformExperienceReplay`` the
        priorities are ignored.
        """
        return
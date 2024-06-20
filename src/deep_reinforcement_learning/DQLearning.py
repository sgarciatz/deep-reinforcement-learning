from typing import Any
import torch
from torch import Tensor
import gymnasium as gym
import random
from deep_reinforcement_learning.QEstimator import QEstimator
from deep_reinforcement_learning.ExperienceMemory import ExperienceMemory
from deep_reinforcement_learning.ActionSelector import ActionSelector
from deep_reinforcement_learning.TrainLogger import TrainLogger
from deep_reinforcement_learning.Experience import Experience
import random
import sys

class DQLearning(object):
    """The implementantion of the Deep Q Learning algorithm.

    Given a dictionary with all the information,initialize the agent
    using the characteristics of the environment and load the training
    hyperparameters.

    Args:
        parameters (dict[str, Any]): The dictionary with all the needed
            information.

    Attributes:
        training_steps (int): The steps (epochs) of the DQL algorithm.
        samples_per_step (int): The number of experiences to sample in
            each iteration of the training process.
        batches (int): The number of batches sampled in each step of the
            training.
        batch_size (int): The number of Experiences that each batch has.
        device (str): For pytorch. It is the identifier of the device to
            use (``'gpu'`` or ``'cpu'``).
        enviroment (Env): The gymnasium-based environment in which the
            agent will be trained on.
        q_estimator (QEstimator): The responsable of estimating the
            Q-value of (state, action) tuples.
        action_selector (ActionSelector): The responsable of selecting
            which action to take based on the policy.
        logger (TrainLogger): This attribute collects information about
            the training to output it.
    """

    def __init__(self, parameters: dict[str, Any]):
        #Set hyperparameters
        self.training_steps: int = parameters["training_steps"]
        self.samples_per_step: int = parameters["h"]
        self.batches: int = parameters["batches"]
        self.batch_size: int = parameters["batch_size"]
        self.updates_per_batch: int = parameters["updates_per_batch"]

        #Set the device for torch
        self.device: str = parameters["device"]

        # Set Components
        self.environment: gym.Env = parameters["env"]

        self.q_estimator: QEstimator = parameters["q_estimator"]
        self.experience_sampler: ExperienceMemory = parameters["memory"]
        self.action_selector: ActionSelector = parameters["action_selector"]

        #Prepare the logging class TrainLogger
        self.logger: TrainLogger = parameters["logger"]

    def _gather_experiences(self):

        """Add ``samples_per_step`` Experiences to the memory.

        Using the current policy (policy_net), fill the replay memory
        buffer. This is the first step of the loop of the DQN
        Algorithm.
        """

        n_experiences = self.samples_per_step
        done = True
        experience = None
        next_state = None
        for i in range(n_experiences):
            if (done):
                seed = random.randint(0, sys.maxsize)
                state, info = self.environment.reset(seed=seed)
            else:
                state = next_state
            raw_state = torch.Tensor(state).to(self.device)
            with (torch.no_grad()):
                q_tar = self.q_estimator.q_estimator(raw_state)
            action = self.action_selector.select_action(q_tar)
            next_state, reward, terminated, truncated, info =\
                self.environment.step(action)
            done = terminated or truncated
            experience = Experience(state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    99)
            self.experience_sampler.add_experience(experience)

    def validate_learning(self, n_validations: int):

        """ Validate the current policy.

        Use the Q estimator network with its current weights to check
        how well the agent performs in a enviroment during
        ``n_validations`` episodes. The key point is that the policy is
        ignored and the option with the highest Q-value is chosen.

        Parameters:
        - n_validations: int = The number of episodes to execute.
        """

        rewards: list[float] = []
        ep_lengths: list[int]= []
        for _ in range(n_validations):
            done: bool = False
            ep_length: int = 0
            ep_reward: float = 0
            state, info = self.environment.reset()
            state: Tensor = Tensor(state).to(self.device)
            while (not done):
                with (torch.no_grad()):
                    q_estimate: Tensor = self.q_estimator.q_estimator(state)
                action: int = self.action_selector.select_action(q_estimate)
                next_state, reward, terminated, truncated, info =\
                self.environment.step(action)
                state: Tensor = torch.Tensor(next_state).to(self.device)
                done = terminated or truncated
                ep_length += 1
                ep_reward += reward
            rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        return sum(rewards) / n_validations, sum(ep_lengths) / n_validations

    def train(self):
        """
        The maing training loop of the DQN agent. The training consists
        of the execution of the following steps:

        Note:
            The pseudocode of the Deep Q Learning algorithm is as
                follows:

            ```
            For each step of training:
                generate h experiences with the current policy
                gather B batches, For each bacth:
                    For each u in batch_updates:
                        calculate the target Q (Qtar) of all experiences
                        calculate the Q estimate (Q^) of all experiences
                        calculate the loss between Qtar and Q^
                        update Q-estimator's parameters
                decrease exploring rate
            ```
        """
        self.logger.print_training_header()
        step_losses: list[float] = []
        batch: list[Experience] = []
        for step in range(1, self.training_steps + 1):
            step_losses = []
            self._gather_experiences()
            for _ in range(self.batches):
                batch = self.experience_sampler.sample_experience(self.batch_size)
                for _ in range(self.updates_per_batch):
                    loss, td_error = self.q_estimator.calculate_q_loss(batch)
                    self.q_estimator.update_q_estimator(loss)
                    self.experience_sampler.update_batch_priorities(
                        batch,
                        td_error)
                    step_losses.append(loss.item())
            reward, ep_length = self.validate_learning(10)
            expl_rate = self.action_selector.exploration_rate
            self.logger.add_training_step(step,
                                          expl_rate,
                                          sum(step_losses) / len(step_losses),
                                          reward,
                                          ep_length)
            self.logger.print_training_step()
            self.action_selector.decay_exploration_rate(step,
                                                        self.training_steps)
            self.q_estimator.update_second_q_estimator(step)
        self.logger.print_training_footer()
        self.q_estimator.pickle_model()

    def test(self, n_validations: int) -> tuple[int, int]:
        """Test the already trained agent. This method is similar to
        validate_learning but it does not use the policy, instead the
        action with the highest Q value is chosen.

        Parameters:
        - n_validations: int = The number of episodes to carry out.
        """
        rewards: list[float] = []
        ep_lengths: list[int] = []
        for _ in range(n_validations):
            done: bool = False
            ep_length: int = 0
            ep_reward: float = 0
            state, info = self.environment.reset()
            state: Tensor = torch.Tensor(state).to(self.device)
            while (not done):
                with (torch.no_grad()):
                    q_estimate: Tensor = self.q_estimator.q_estimator(state)
                action: int = q_estimate.argmax().item()
                next_state, reward, terminated, truncated, info =\
                    self.environment.step(action)
                state = torch.Tensor(next_state).to(self.device)
                done = terminated or truncated
                ep_length += 1
                ep_reward += reward
            rewards.append(ep_reward)
            ep_lengths.append(ep_length)
        return sum(rewards) / n_validations, sum(ep_lengths) / n_validations

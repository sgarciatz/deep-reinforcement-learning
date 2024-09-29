from typing import Any, SupportsFloat
import torch
from torch import Tensor
import gymnasium as gym
import random
from deep_reinforcement_learning.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from deep_reinforcement_learning.QEstimator import QEstimator
from deep_reinforcement_learning.ExperienceMemory import ExperienceMemory
from deep_reinforcement_learning.ActionSelector import ActionSelector
from deep_reinforcement_learning.TrainLogger import TrainLogger
from deep_reinforcement_learning.Experience import Experience
import random
import sys
import time


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
        self.experience_memory: ExperienceMemory = parameters["memory"]
        self.action_selector: ActionSelector = parameters["action_selector"]

        #Prepare the logging class TrainLogger
        self.logger: TrainLogger = parameters["logger"]

        self.done: bool = True
        self.experience: Experience | None = None
        self.next_state: torch.Tensor | None = None

    def _gather_experiences(self,
                            n_experiences: int | None = None):

        """Add ``samples_per_step`` Experiences to the memory.

        Using the current policy (policy_net), fill the replay memory
        buffer. This is the first step of the loop of the DQN
        Algorithm.
        """
        if (n_experiences == None):
            n_experiences = self.samples_per_step
            if (len(self.experience_memory.experience_buffer) == 0):
                n_experiences = self.batch_size * self.batches

        for i in range(n_experiences):
            if (self.done):
                seed = random.randint(0, sys.maxsize)
                state, info = self.environment.reset(seed=seed)
            else:
                state = self.next_state
            with (torch.no_grad()):
                q_tar = self.q_estimator.policy_qnet(state)
            action: torch.Tensor = torch.tensor(
                self.action_selector.select_action(q_tar),
                dtype=torch.int32
            )
            self.next_state, reward, terminated, truncated, info =\
                self.environment.step(action)
            done = terminated or truncated
            self.experience = Experience(state,
                                         action,
                                         reward,
                                         self.next_state,
                                         done,
                                         99.0)
            self.experience_memory.add_experience(self.experience)

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
                    q_estimate: Tensor = self.q_estimator.policy_qnet(state)
                # action: int = self.action_selector.select_action(q_estimate)
                best_action: int = q_estimate.argmax(dim=1).item()
                chosen_action: int = self.action_selector.select_action(q_estimate)
                # print(torch.round(q_estimate, decimals=2), best_action, chosen_action)
                next_state, reward, terminated, truncated, info =\
                self.environment.step(action=best_action)
                state: Tensor = torch.Tensor(next_state).to(self.device)
                done = terminated or truncated
                ep_length += 1
                ep_reward += reward # type: ignore
            # print()
            rewards.append(ep_reward)
            ep_lengths.append(ep_length)
        avg_reward = sum(rewards) / n_validations
        avg_ep_length = sum(ep_lengths) / n_validations
        self.done = True
        self.experience = None
        self.next_state = None
        return avg_reward, avg_ep_length


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
        step_losses: list[float] = []
        batch: list[Experience] = []
        for step in range(1, self.training_steps + 1):
            # e, b, l, u, p = [], [], [], [], []
            step_losses = []
            # start_time = time.time_ns()
            self._gather_experiences()
            # e.append((time.time_ns() -start_time)*10e-9)
            for _ in range(self.batches):
                # start_time = time.time_ns()
                batch = self.experience_memory.sample_experience(
                    self.batch_size)
                # b.append((time.time_ns() -start_time)*10e-9)
                for _ in range(self.updates_per_batch):
                    # start_time = time.time_ns()
                    loss, td_error =\
                        self.q_estimator.calculate_q_loss(batch)
                    # l.append((time.time_ns() -start_time)*10e-9)
                    # start_time = time.time_ns()
                    self.q_estimator.update_policy_qnet(loss)
                    # u.append((time.time_ns() -start_time)*10e-9)
                    # start_time = time.time_ns()
                    self.experience_memory.update_batch_priorities(
                        batch,
                        td_error)
                    # p.append((time.time_ns() -start_time)*10e-9)
                    step_losses.append(loss.item())
            reward, ep_length = self.validate_learning(10)
            expl_rate = self.action_selector.exploration_rate
            self.logger.add_training_step(
                step,
                expl_rate,
                sum(step_losses) / len(step_losses),
                reward,
                ep_length)
            self.action_selector.decay_exploration_rate(step,
                                                        self.training_steps)
            self.q_estimator.update_target_qnet(step)
        #     print()
        #     print(f"Experience generation time {sum(e) / len(e)}")
        #     print(f"Batch preparation time {sum(b) / len(b)}")
        #     print(f"Loss calculation time {sum(l) / len(l)}")
        #     print(f"Q_estimator updating time {sum(u) / len(u)}")
        #     print(f"Priorities updating time {sum(p) / len(p)}")
        #     print(f"Training Step time {(sum(e) / len(e)) + (sum(b) / len(b)) + (sum(l) / len(l)) + (sum(u) / len(u)) + (sum(p) / len(p))}")
        # exit()
        self.q_estimator.pickle_model()

    def test(self, n_validations: int) -> tuple[float, float]:
        """Test the already trained agent. This method is similar to
        validate_learning but it does not use the policy, instead the
        action with the highest Q value is chosen.

        Parameters:
        - n_validations: int =

        Args:
            n_validations (int): The number of episodes to carry out.

        Returns:
            tuple[float, float]: The average reward and episode length.
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
                    q_estimate: Tensor = self.q_estimator.policy_qnet(state)
                action: int = int(q_estimate.argmax().item())
                next_state, reward, terminated, truncated, info =\
                    self.environment.step(action)
                state = torch.Tensor(next_state).to(self.device)
                done = terminated or truncated
                ep_length += 1
                ep_reward += reward # type: ignore
            rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            mean_reward: float = sum(rewards) / n_validations
            mean_ep_length: float = sum(ep_lengths) / n_validations
        return mean_reward, mean_ep_length
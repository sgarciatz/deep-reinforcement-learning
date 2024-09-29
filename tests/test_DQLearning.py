from pathlib import Path
import unittest
import torch
import numpy as np
import deep_reinforcement_learning as drl
import gymnasium as gym
from typing import Any, SupportsFloat
import random
from deep_reinforcement_learning.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from deep_reinforcement_learning.ActionSelector import ActionSelector

class SimplestEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            n=2,
            start=0,
            seed=42
        )
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=np.array(
                [0, 0],
                dtype=np.int64
            ),
            high=np.array(
                [20, 20],
                dtype=np.int64
            ),
            shape=(2,),
            seed=42
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if (random.random() >= 0.5):
            self.target: int = random.randint(10, 20)
            self.agent: int = random.randint(0, 10)
        else:
            self.target: int = random.randint(0, 10)
            self.agent: int = random.randint(10, 20)
        self.steps = 0
        state: torch.Tensor = torch.tensor(
            [self.target, self.agent],
            dtype=torch.float32
        )
        return state, {}
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == 0):
            self.agent += 1
        elif ( action == 1):
            self.agent -=1
        state: torch.Tensor = torch.tensor(
            [self.target, self.agent],
            dtype=torch.float32
        )
        is_outside: bool = (self.agent < 0) or (self.agent > 20)
        is_finished: bool = self.agent == self.target
        took_too_long: bool = self.steps > 50
        reward: torch.Tensor = torch.tensor(0.0, dtype=torch.float32)
        done: torch.Tensor = torch.tensor(False, dtype=torch.bool)
        truncated: torch.Tensor = torch.tensor(False, dtype=torch.bool)
        if (is_outside):
            done = torch.tensor(True, dtype=torch.bool)
        elif (is_finished):
            reward: torch.Tensor = torch.tensor(1.0, dtype=torch.float32)
            done: torch.Tensor = torch.tensor(True, dtype=torch.bool)
        elif (took_too_long):
            truncated = torch.tensor(True, dtype=torch.bool)
        self.steps += 1
        return state, reward, done, truncated, {"hola": "hola"}

class ToyEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            4,
            start=0,
            seed=42)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=np.array(
                [[0,0], [0,0]],
                dtype=int),
            high=np.array(
                [[2,2], [2,2]],
                dtype=int),
            shape=(2, 2),
            dtype=np.int32)
        self.steps: int = 0
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.target: list[int] = [2, 2]
        self.agent: list[int] = [0, 0]
        self.steps = 0
        state: torch.Tensor = torch.tensor(
            [self.target[0], self.target[1], self.agent[0], self.agent[1]],
            dtype=torch.float32
        )
        return state, {}

    def step(self, action: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Apply the action chosen by the agent
        if (action == 0): # up
            self.agent[0] += 1
        elif (action == 1): #right
            self.agent[1] += 1
        elif (action == 2): #down
            self.agent[0] -= 1
        elif (action == 3): #left
            self.agent[1] -= 1
        # Get the observation
        obs: torch.Tensor = torch.tensor(
             [self.target[0], self.target[1], self.agent[0], self.agent[1]],
             dtype=torch.float32
        )
        # Check where the agent is
        is_outside: bool = (self.agent[0] < 0)
        is_outside = is_outside or (self.agent[0] > 2)
        is_outside = is_outside or (self.agent[1] < 0)
        is_outside = is_outside or (self.agent[1] > 2)
        is_finished: bool = self.agent[0] == self.target[0]
        is_finished = is_finished and (self.agent[1] == self.target[1])
        took_too_much: bool = self.steps > 20
        if (is_outside):
            # Apply penalty
            reward: torch.Tensor = torch.tensor(0.0, dtype=torch.float32)
            done : torch.Tensor = torch.tensor(True, dtype=torch.bool)
            truncated : torch.Tensor = torch.tensor(False, dtype=torch.bool)
        elif (is_finished):
            reward: torch.Tensor = torch.tensor(1.0, dtype=torch.float32)
            done : torch.Tensor = torch.tensor(True, dtype=torch.bool)
            truncated : torch.Tensor = torch.tensor(False, dtype=torch.bool)
        elif (took_too_much):
            reward: torch.Tensor = torch.tensor(0.0, dtype=torch.float32)
            done : torch.Tensor = torch.tensor(False, dtype=torch.bool)
            truncated : torch.Tensor = torch.tensor(True, dtype=torch.bool)
        else:
            reward: torch.Tensor = torch.tensor(0.0, dtype=torch.float32)
            done : torch.Tensor = torch.tensor(False, dtype=torch.bool)
            truncated : torch.Tensor = torch.tensor(False, dtype=torch.bool)
        self.steps += 1
        return obs, reward, done, truncated, {"hola": "hola"}

class test_DQLearning(unittest.TestCase):

    def test_DQLearning(self):
        """Test that DQL learns in a toy envionrment
        """
        return 
        device: str = "cpu"
        env = SimplestEnv()
        # Create q_networks
        policy_q_network = drl.QNetwork(2,
                                    2,
                                    [2, [2, 64], [64, 64], [64,2], 2],
                                    device)
        target_q_network = drl.QNetwork(2,
                                    2,
                                    [2, [2, 64], [64, 64], [64,2], 2],
                                    device)
        target_q_network.load_state_dict(policy_q_network.state_dict())

        # Configure QEstimator
        variation = "simple"
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(policy_q_network.parameters(),
                                    lr=1e-4)

        gamma = 0.8 #exploration rate
        update_policy = "replace"
        update_param = 10 # number of steps

        q_estimator = drl.QEstimator(policy_q_network,
                                optimizer,
                                loss_fn,
                                gamma,
                                device,
                                update_policy,
                                update_param,
                                None,#target_q_network,
                                variation,
                                "what")
        # Configure ActionSelector
        temperature = 1.0
        start_exploration_rate = 0.9
        end_exploration_rate = 0.0
        decay_strat = "linear"
        decay_rate = 1.0

        boltzmann_policy = drl.BoltzmannPolicy(temperature)
        eps_greedy_policy: EpsilonGreedyPolicy = EpsilonGreedyPolicy(e=start_exploration_rate)




        action_selector: ActionSelector = ActionSelector(
            eps_greedy_policy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_exploration_rate,
            end_exploration_rate=end_exploration_rate,
            decay_rate=None)
        # Configure ExperienceMemory
        buffer_size = 51200
        epsilon = 1e-5
        alpha = 1.0
        experience_memory = drl.PrioritizedExperienceReplay(
            buffer_max_size=buffer_size,
            device=device,
            epsilon=epsilon,
            alpha=alpha)
        # Configure TrainLogger

        output_dir = Path("./", "training/", "runs")
        training_name: Path = Path("what")
        train_logger = drl.TrainLogger(output_dir,
                                       training_name,
                                       {})
        configuration_dict = {}



        configuration_dict["id"] = "what"
        configuration_dict["env"] = env
        configuration_dict["device"] = device

        configuration_dict["logger"] = train_logger

        configuration_dict["training_steps"] = 100
        configuration_dict["batches"] = 8
        configuration_dict["batch_size"] = 128
        configuration_dict["updates_per_batch"] = 4
        configuration_dict["h"] = 1024
        configuration_dict["q_estimator"] = q_estimator
        configuration_dict["action_selector"] = action_selector
        configuration_dict["memory"] = experience_memory

        # Initialize agent
        agent = drl.DQLearning(configuration_dict)
        agent.train()
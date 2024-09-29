import unittest
import torch
import random
from deep_reinforcement_learning.QEstimator import QEstimator
from deep_reinforcement_learning.QDuelingNetwork import QDuelingNetwork
from deep_reinforcement_learning.Experience import Experience


class test_QEstimator(unittest.TestCase):

    def test_initialization(self):
        """check that the QEstimator is initialized correctly.
        """
        n_obs = 32
        n_act = 5
        layers = [32, [32, 64], [64, 128], [128, 64], [64, 32], 32]
        gamma = 0.9
        device = "cpu"
        update_policy = "replace"
        update_freq = 50
        variation = "ddqn"
        policy_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net.load_state_dict(policy_net.state_dict())
        loss_fn = torch.nn.HuberLoss()
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(policy_net.parameters(),
                                      learning_rate,
                                      amsgrad=True)
        q_estimator = QEstimator(policy_net,
                                 optimizer,
                                 loss_fn,
                                 gamma,
                                 device,
                                 update_policy,
                                 update_freq,
                                 target_net,
                                 variation)
        self.assertEqual(policy_net, q_estimator.policy_qnet)
        self.assertEqual(target_net, q_estimator.target_qnet)
        self.assertEqual(optimizer, q_estimator.optimizer)
        self.assertEqual(loss_fn, q_estimator.loss_fn)
        self.assertEqual(gamma, q_estimator.gamma)
        self.assertEqual(device, q_estimator.device)
        self.assertEqual(update_policy, q_estimator.update_policy)
        self.assertEqual(variation, q_estimator.variation)

    def test_replace_update(self):
        """Check that when update_second_q_estimator is called,
        both state dicts are the same.
        """
        n_obs = 32
        n_act = 5
        layers = [32, [32, 64], [64, 128], [128, 64], [64, 32], 32]
        gamma = 0.9
        device = "cpu"
        update_policy = "replace"
        update_freq = 50
        variation = "dqn"
        policy_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net.load_state_dict(policy_net.state_dict())
        loss_fn = torch.nn.HuberLoss()
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(policy_net.parameters(),
                                      learning_rate,
                                      amsgrad=True)
        q_estimator = QEstimator(policy_net,
                                 optimizer,
                                 loss_fn,
                                 gamma,
                                 device,
                                 update_policy,
                                 update_freq,
                                 target_net,
                                 variation)
        n_experiences = 100
        random.seed(24)
        batch: list[Experience] = []
        for index in range(n_experiences):
            batch.append(
                Experience(
                    state=torch.tensor(
                        [float(random.randrange(0, 9)) for _ in range(n_obs)],
                        dtype=torch.float32),
                    action=torch.tensor(
                        random.randint(0, 4),
                        dtype=torch.int32),
                    reward=torch.tensor(
                        random.randrange(-1, 1)),
                    next_state=torch.tensor(
                        [float(random.randrange(0, 9)) for _ in range(n_obs)],
                        dtype=torch.float32),
                    done=torch.tensor(False, dtype=torch.bool),
                    priority=torch.tensor(1000.0, dtype=torch.float32)
                )
            )

        loss, _ = q_estimator.calculate_q_loss(batch)
        self.assertEqual(q_estimator.policy_qnet.state_dict().__str__(),
                         q_estimator.target_qnet.state_dict().__str__())
        q_estimator.update_policy_qnet(loss)
        self.assertNotEqual(q_estimator.policy_qnet.state_dict().__str__(),
                            q_estimator.target_qnet.state_dict().__str__())
        q_estimator.update_target_qnet(update_freq)
        self.assertEqual(
            q_estimator.policy_qnet.state_dict().__str__(),
            q_estimator.target_qnet.state_dict().__str__())

    def test_polyak_update(self):
        """Check that when update_second_q_estimator is called,
        both state dicts are the same.
        """
        n_obs: int = 32
        n_act: int = 5
        layers: list = [32, [32, 64], [64, 128], [128, 64], [64, 32], 32]
        gamma: float = 0.9
        device: str = "cpu"
        update_policy: str = "polyak"
        update_freq: int = 50
        variation: str = "dqn"
        policy_net: QDuelingNetwork = QDuelingNetwork(
            n_obs,
            n_act,
            layers,
            device)
        target_net: QDuelingNetwork = QDuelingNetwork(
            n_obs,
            n_act,
            layers,
            device)
        target_net.load_state_dict(policy_net.state_dict())
        loss_fn: torch.nn.HuberLoss = torch.nn.HuberLoss()
        learning_rate: float = 1e-4
        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            policy_net.parameters(),
            learning_rate,
            amsgrad=True)
        q_estimator: QEstimator = QEstimator(
            policy_qnet=policy_net,
            optimizer=optimizer,
            loss_fn=loss_fn,
            gamma=gamma,
            device=device,
            update_param=update_policy,
            update_policy=update_freq,
            target_qnet=target_net,
            variation=variation)
        n_experiences:int = 100
        random.seed(24)

        batch: list[Experience] = []
        for _ in range(n_experiences):
            batch.append(
                Experience(
                    state=torch.tensor(
                        [float(random.randrange(0, 9)) for _ in range(n_obs)],
                        dtype=torch.float32),
                    action=torch.tensor(
                        random.randint(0, 4),
                        dtype=torch.int32),
                    reward=torch.tensor(
                        random.randrange(-1, 1),
                        dtype=torch.int32),
                    next_state=torch.tensor(
                        [float(random.randrange(0, 9)) for _ in range(n_obs)],
                        dtype=torch.float32),
                    done=torch.tensor(False, dtype=torch.bool),
                    priority=torch.tensor(1000.0, dtype=torch.float32)
                )
            )
        loss: torch.Tensor = q_estimator.calculate_q_loss(batch)[0]
        self.assertEqual(
            q_estimator.policy_qnet.state_dict().__str__(),
            q_estimator.target_qnet.state_dict().__str__())
        q_estimator.update_policy_qnet(loss)
        self.assertNotEqual(
            q_estimator.policy_qnet.state_dict().__str__(),
            q_estimator.target_qnet.state_dict().__str__())
        q_estimator.update_target_qnet(update_freq)
        q_estimator.update_target_qnet(update_freq)
        self.assertNotEqual(
            q_estimator.policy_qnet.state_dict().__str__(),
            q_estimator.target_qnet.state_dict().__str__())

if __name__ == '__main__':
    unittest.main()
import unittest
import torch
from deep_reinforcement_learning.QNetwork import QNetwork
import random


class test_QNetwork(unittest.TestCase):

    def test_create_q_network(self):
        """Test that the network is created accordingly to the
        expecifications provided.
        """

        n_obs = 32
        n_act = 4
        layers = [64, [64, 128], [128, 64], 64]
        device = "cpu"
        net = QNetwork(n_obs, n_act, layers, device)
        expected_layers = len(layers) * 2 -1
        actual_layers = len(net.layer_stack)
        self.assertEqual(expected_layers, actual_layers)
        input_layer = net.layer_stack[0]
        self.assertEqual(n_obs, input_layer.in_features)
        self.assertEqual(input_layer.out_features,
                        layers[0])
        for index in range(2, actual_layers-1, 2):
            actual_layer = net.layer_stack[index]
            expected_in_features = layers[index//2][0]
            expected_out_features = layers[index//2][1]
            self.assertEqual(actual_layer.in_features, expected_in_features)
            self.assertEqual(actual_layer.out_features, expected_out_features)

        output_layer = net.layer_stack[-1]
        self.assertEqual(output_layer.in_features, layers[-1])
        self.assertEqual(output_layer.out_features, n_act)

    def test_forward(self):
        """Test that the forward output is consistent with the action
        space.
        """
        n_obs = 32
        n_act = 4
        layers = [64, [64, 128], [128, 64], 64]
        device = "cpu"
        net = QNetwork(n_obs, n_act, layers, device)
        x = torch.tensor([i for i in range(32)], dtype=torch.float32)
        y = net(x)
        self.assertEqual(y.shape[1], n_act)

    def test_batch_forward(self):
        """Test that the forward output is consistent with the action
        space when a batch is feeded.
        """
        n_obs = 32
        n_act = 4
        layers = [64, [64, 128], [128, 64], 64]
        device = "cpu"
        net = QNetwork(n_obs, n_act, layers, device)
        x = torch.tensor([i for i in range(32)], dtype=torch.float32)
        batch = x.repeat(128, 1)
        y = net(batch)
        self.assertEqual(y.shape[0], batch.shape[0])
        self.assertEqual(y.shape[1], n_act)

    def test_fitness(self):
        """Check that weights can be configurated so that the network
        fits static data. For example, the net must fit the following
        function, given batches of observations of 3*64=192 items within
        the [0, 1] range, it shall output the sum of the items % 64
        """
        n_obs: int = 10
        n_act: int = 4
        layers: list[list[int] | int] = [n_obs,
                        [n_obs, 64],
                        [64, 32],
                        [32, n_act],
                        n_act]
        qnet: QNetwork = QNetwork(n_observations=n_obs,
                                  n_actions=n_act,
                                  layers=layers)
        device: str = "cpu"

        batch_size: int = 100
        obs_batch: torch.Tensor = torch.rand((n_obs, batch_size),
                                               dtype=torch.float).T

        target_qvalues: torch.Tensor = torch.sum(obs_batch, dim=1)

        target_qvalues: torch.Tensor = target_qvalues.int() % n_act
        target_qvalues = target_qvalues.repeat((1,1)).float()
        actions_aux: list = []
        for i in range(1, n_act+1):
            actions_aux.append(i)

        actions: torch.Tensor = torch.tensor(actions_aux, dtype=torch.float)
        actions = actions.repeat((1, 1)).float()


        target_qvalues = torch.matmul(target_qvalues.T, actions)
        loss_fn: torch.nn.HuberLoss = torch.nn.HuberLoss()
        learning_rate: float = 1e-3
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(qnet.parameters(),
                                                             lr=learning_rate)
        epochs: int = 100
        losses: list[float] = []
        for epoch in range(epochs):
            pred_qvalues: torch.Tensor = qnet(obs_batch)
            loss: torch.Tensor = loss_fn(pred_qvalues, target_qvalues)
            losses.append(float(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step: int = 10
        for i in range(step, len(losses), step):
            self.assertGreater(losses[i-10], losses[i])

    def test_fitness2(self) -> None:
        """Check that the network is able to learn a function that given
        two numbers within the [0, 9] range returns [1, 0] if the first
        is greater, [1, 1] if they are equal and [0, 1] if the second is
        greater.
        """
        n_obs: int = 2
        n_act: int = 2

        layers = [64, [64, 64], 64]
        qnet: QNetwork = QNetwork(n_observations=n_obs,
                            n_actions=n_act,
                            layers=layers,
                            device="cpu")
        batch_size: int = 100



        loss_fn: torch.nn.HuberLoss = torch.nn.HuberLoss()
        learning_rate: float = 1e-3
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(qnet.parameters(),
                                                             lr=learning_rate)
        losses: list = []
        for _ in range(300):
            x = []
            y = []
            for _ in range(batch_size):
                a = float(random.randint(0, 9))
                b = float(random.randint(0, 9))
                x.append([a,b])
                if (a > b):
                    y.append([1.0, 0.0])
                elif (a < b):
                    y.append([0.0, 1.0])
                else:
                    y.append([1.0, 1.0])
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            y_pred: torch.Tensor = qnet(x)
            loss: torch.Tensor = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss)
            optimizer.step()
        self.assertGreater(losses[0], losses[-1])
if __name__ == '__main__':
    unittest.main()
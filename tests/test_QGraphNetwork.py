import unittest
import torch
from deep_reinforcement_learning.QGraphNetwork import QGraphNetwork
from collections import deque
import random

class test_QGraphNetwork(unittest.TestCase):

    def test_fitness2(self) -> None:
        n_obs: int = 1
        n_act: int = 2
        device: str = "cpu"
        qnet: QGraphNetwork = QGraphNetwork(actions=n_act,
                                            node_observations=n_obs,
                                            edge_observations=0,
                                            graph_convolutions=1)
        batch_size: int = 100
        loss_fn: torch.nn.HuberLoss = torch.nn.HuberLoss()
        learning_rate: float = 1e-4
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(qnet.parameters(),
                                                             lr=learning_rate)
        losses: list = []
        for _ in range(300):
            x = []
            y = []
            for _ in range(batch_size):
                a = float(random.randint(0, 9))
                b = float(random.randint(0, 9))
                x.append([[a],[b]])
                if (a > b):
                    y.append([1.0, 0.0])
                elif (a < b):
                    y.append([0.0, 1.0])
                else:
                    y.append([1.0, 1.0])
            x = (torch.tensor(x, dtype=torch.float32),
                 torch.tensor(0.0, dtype=torch.float32),
                 torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32))
            y = torch.tensor(y, dtype=torch.float32)
            y_pred: torch.Tensor = qnet(x)
            loss: torch.Tensor = loss_fn(y_pred, y)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertGreater(losses[0],
                           losses[-1],
                           msg="Loss did not decrease.")
if __name__ == '__main__':
    unittest.main()
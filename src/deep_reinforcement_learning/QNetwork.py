import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):


    """
    This network is used to estimate the Q function by means of
    reducing the temporal difference error during its training
    pahse
    """


    def __init__(self,
                 n_observations: int,
                 n_actions: int,
                 layers: list,
                 device: str = "cpu"):
        """Create the NN stacking the layers

        Args:
            n_observations (int): The number of observations and size of
            the input layer.
            n_actions (int): The number of different actions and the
            size of the output layer.
            layers (list): A list with the number of Linear layers and
            their number of neurons.
            device (str, optional): The device for pytorch (cuda or cpu).
            Defaults to "cpu".
        """
        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        input_layer = nn.Linear(n_observations, layers[0])
        output_layer = nn.Linear(layers[-1],
                                      n_actions)
        hidden_layers = []
        for layer in layers[1:-1]:
            hidden_layers.append(nn.Linear(layer[0], layer[1]))
            hidden_layers.append(nn.ReLU())
        self.layer_stack = nn.Sequential(
            input_layer,
            nn.ReLU(),
            *hidden_layers,
            output_layer).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed input data into the Q-Network.

        Args:
            x (torch.Tensor): A minibatch of states.

        Returns:
            _type_: The resulting logits.
        """
        logits = self.layer_stack(x.repeat(1,1))
        return logits


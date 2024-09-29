import torch
from torch.nn.functional import softmax, kl_div
from torch.nn.modules.loss import MSELoss, HuberLoss, L1Loss, CrossEntropyLoss
import torch.nn as nn
import torch.optim as optim
import numpy as np
from deep_reinforcement_learning.Experience import Experience
from deep_reinforcement_learning.Policy import Policy

ValidLoss = MSELoss | HuberLoss | L1Loss | CrossEntropyLoss

class QEstimator(object):
    """A estimator of Q-values for (state, action) tuples.

    This class' purpose is to manage the Q-estimator of a DRL agent. It
    may be implemented with a single Q-Network or with multiples,
    depending on the variation.

    Attributes:
        device (str): For pytorch. It is the identifier of the device to
            use (``'gpu'`` or ``'cpu'``).
        q_estimator (nn.Module): The Q-Network used to estimate the
            Q-function.
        n_actions (int): The size of the action space.
        optimizer (optim.Optimizer): The optimizer used to adjust
            the weights of q_estimator.
        gamma (float): is the temporal discount factor. γ. Defaults to
            0.9.
        loss_fn: The function used to calculate the loss.

        update_policy (str): The policy followed to update the second
            network: ``replace``or ``polyak``.
        update_param (int | float): The polyak factor or the replace
            period.
        second_q_estimator (nn.Module): The second Q-Network used in
            some DQL variations (target net, Double DQN...).
    """

    def __init__(self,
                 policy_qnet: nn.Module,
                 optimizer: optim.Optimizer,
                 loss_fn: ValidLoss,
                 gamma: float = 0.9,
                 device: str = "cpu",
                 update_policy: str = "replace",
                 update_param: int|float = 50,
                 target_qnet: nn.Module | None = None,
                 variation: str = "ddqn",
                 output_path: str = "../models/modelito.pt"
                 ):
        self.device: str = device
        self.policy_qnet: nn.Module = policy_qnet
        self.n_actions: int = self.policy_qnet.n_actions
        self.optimizer: optim.Optimizer = optimizer
        self.loss_fn: ValidLoss = loss_fn
        self.gamma: torch.Tensor = torch.tensor(gamma).to(self.device)
        self.update_policy: str = update_policy
        self.update_param: int|float = update_param
        self.target_qnet: nn.Module | None = target_qnet
        if (self.target_qnet is not None):
            self.target_qnet.load_state_dict(
                self.policy_qnet.state_dict())
        self.variation: str = variation
        self.output_path: str = output_path

    def calculate_q_loss(self,
                         batch: list[Experience]
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a batch, calculate the loss using the given loss_fn.

        Args:
            batch (list[Experience]): The batch of Experiences.

        Returns:
            tuple[Tensor, Tensor]: The loss and the temporal difference
                error.
        """
        states: torch.Tensor = torch.stack(
            tuple([e.state for e in batch])).to(self.device)
        actions: torch.Tensor = torch.stack(
            tuple([e.action for e in batch]),).to(
                dtype=torch.int64,
                device=self.device)
        next_states: torch.Tensor = torch.stack(
            tuple([e.next_state for e in batch])).to(self.device)
        rewards: torch.Tensor = torch.stack(
            tuple([e.reward for e in batch])).to(self.device)
        dones: torch.Tensor = torch.stack(
            tuple([e.done for e in batch])).to(self.device)
        # Obtain the estimated Q values of the initial state.
        q_preds: torch.Tensor = self.policy_qnet(states)
        q_preds = torch.flatten(q_preds.gather(1, actions.repeat((1,1)).T))
        with (torch.no_grad()):
            if (self.target_qnet is not None):
                if (self.variation == "ddqn"):
                    # Obtain the Q values of the state to which the agent
                    # transitions.
                    q_tars_next_idx: torch.Tensor = self.target_qnet(next_states)
                    q_tars_next_idx = q_tars_next_idx.max(dim=1).indices
                    q_tars_next_idx = q_tars_next_idx.unsqueeze(dim=1)
                    q_tars_next: torch.Tensor = self.policy_qnet(next_states)
                    q_tars_next = q_tars_next.gather(1, q_tars_next_idx)
                    q_tars_next = q_tars_next.flatten()
                else:
                    q_tars_next = self.target_qnet(next_states)
                    q_tars_next = q_tars_next.max(dim=1).values
            else:
                q_tars_next = self.policy_qnet(next_states)

                q_tars_next = q_tars_next.max(dim=1).values
        q_tars: torch.Tensor = rewards + ( dones * self.gamma * q_tars_next)
        loss: torch.Tensor = self.loss_fn(q_preds, q_tars)
        td_error: torch.Tensor = torch.abs(q_tars - q_preds)
        return loss, td_error

    def update_policy_qnet(self,
                           loss: torch.Tensor) -> None:
        """Updates the primary q_estimator given the loss and the
        optimizer.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_qnet(self, step: int) -> None:
        """Updates the secondary q estimator (polyak or replace).

        Parameters:
        - step (int): the current step of the training.
        """
        if (self.target_qnet is None):
            return
        if (self.update_policy == "replace"\
            and step % self.update_param == 0):
            print("HOLA")
            self.target_qnet\
                    .load_state_dict(self.policy_qnet.state_dict())
        if (self.update_policy == "polyak"):
            q_dict = self.policy_qnet.state_dict()
            second_q_dict = self.target_qnet.state_dict()
            for key in q_dict:
                second_q_dict[key] = q_dict[key]*self.update_param\
                     + second_q_dict[key]*(1-self.update_param)
            self.target_qnet.load_state_dict(second_q_dict)

    def pickle_model(self):
        """Pickle the resulting model."""
        torch.save(self.policy_qnet.state_dict(), self.output_path)

    def load_model(self, device: str = "cpu"):
        """Load pickled model.

        Args:
            device (str, optional): ``cpu`` or ``gpu``. Used to let
            torch know what device to use. Defaults to "cpu".
        """
        self.target_qnet = None
        self.policy_qnet.load_state_dict(
            torch.load(self.output_path,
                       map_location=torch.device(device)))


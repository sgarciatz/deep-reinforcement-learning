import torch
from torch import Tensor
from torch.nn.functional import softmax, kl_div
import torch.nn as nn
import torch.optim as optim
import numpy as np
from deep_reinforcement_learning.Experience import Experience


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
                 q_estimator: nn.Module,
                 optimizer: optim.Optimizer,
                 loss_fn,
                 gamma: float = 0.9,
                 device: str = "cpu",
                 update_policy: str = "replace",
                 update_param: int|float = 50,
                 second_q_estimator: nn.Module = None,
                 variation: str = "ddqn",
                 output_path: str = "../models/modelito.pt"
                 ):
        self.device = device
        self.q_estimator = q_estimator
        self.n_actions = self.q_estimator.n_actions
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = torch.tensor(gamma)
        self.update_policy = update_policy
        self.update_param = update_param
        self.second_q_estimator = second_q_estimator
        if (self.second_q_estimator is not None):
            self.second_q_estimator.load_state_dict(
                self.q_estimator.state_dict())
        self.variation = variation
        self.output_path = output_path

    def calculate_q_loss(self,
                         batch: list[Experience]) -> tuple[Tensor, Tensor]:
        """Given a batch, calculate the loss using the given loss_fn.

        Args:
            batch (list[Experience]): The batch of Experiences.

        Returns:
            tuple[Tensor, Tensor]: The loss and the temporal difference
                error.
        """
        states = torch.tensor([e.state for e in batch],
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in batch],
                               dtype=torch.int64).to(self.device)

        next_states = torch.tensor([e.next_state for e in batch],
                                   dtype=torch.float32).to(self.device)
        rewards = torch.tensor([e.reward for e in batch],
                                dtype=torch.float32).to(self.device)
        dones = torch.tensor([e.done for e in batch],
                                dtype=torch.float32).to(self.device)
        # Obtain the estimated Q values of the initial state.
        q_preds: Tensor = self.q_estimator(states)
        q_preds_kl: Tensor = softmax(
            (q_preds - torch.min(q_preds))\
                / (torch.max(q_preds) - torch.min(q_preds)),
            dim=1)
        q_preds = torch.flatten(q_preds.gather(1, actions.repeat((1,1)).T))
        with (torch.no_grad()):
            if (self.second_q_estimator is not None):
                if (self.variation == "ddqn"):
                    # Obtain the Q values of the state to which the agent
                    # transitions.
                    q_tar_next_idx = self.second_q_estimator(next_states)\
                                        .max(dim=1).indices.unsqueeze(dim=1)
                    q_tar_next = self.q_estimator(next_states)
                    q_tar_next_kl = q_tar_next
                    q_tar_next = q_tar_next.gather(1, q_tar_next_idx).flatten()
                else:
                    q_tar_next = self.second_q_estimator(next_states)
                    q_tar_next_kl = q_tar_next
                    q_tar_next = q_tar_next_kl.max(dim=1).values
            else:
                q_tar_next = self.q_estimator(next_states)
                q_tar_next_kl = q_tar_next

                q_tar_next = q_tar_next.max(dim=1).values
        q_tars: Tensor = rewards + ( dones * self.gamma * q_tar_next)

        future_reward = (dones * self.gamma).repeat((1, 1)).T * q_tar_next_kl
        present_reward = rewards.repeat((future_reward.shape[1], 1)).T

        q_tars_kl: Tensor = present_reward + future_reward
        q_tars_kl = softmax(
            (((q_tars_kl - torch.min(q_tars_kl))\
            / (torch.max(q_tars_kl) - torch.min(q_tars_kl))).T).T)
        kl_divergence = kl_div(q_preds_kl.log(),
                               q_tars_kl,
                               reduction="mean")
        loss = self.loss_fn(q_preds, q_tars)
        td_error = torch.abs(q_tars - q_preds)
        return loss, td_error, kl_divergence

    def update_q_estimator(self, loss) -> None:
        """Updates the primary q_estimator given the loss and the
        optimizer.
        """
        loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.q_estimator.parameters(),
        #     1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_second_q_estimator(self, step: int) -> None:
        """Updates the secondary q estimator (polyak or replace).

        Parameters:
        - step (int): the current step of the training.
        """
        if (self.second_q_estimator is None):
            return
        if (self.update_policy == "replace"\
            and step % self.update_param == 0):
            self.second_q_estimator\
                    .load_state_dict(self.q_estimator.state_dict())
        if (self.update_policy == "polyak"):
            q_dict = self.q_estimator.state_dict()
            second_q_dict = self.second_q_estimator.state_dict()
            for key in q_dict:
                second_q_dict[key] = q_dict[key]*self.update_param\
                     + second_q_dict[key]*(1-self.update_param)
            self.second_q_estimator.load_state_dict(second_q_dict)

    def pickle_model(self):
        """Pickle the resulting model."""
        torch.save(self.q_estimator.state_dict(), self.output_path)

    def load_model(self, device: str = "cpu"):
        """Load pickled model.

        Args:
            device (str, optional): ``cpu`` or ``gpu``. Used to let
            torch know what device to use. Defaults to "cpu".
        """
        self.second_q_estimator = None
        self.q_estimator.load_state_dict(
            torch.load(self.output_path,
                       map_location=torch.device(device)))


import os
import json
from src.QEstimator import QEstimator
from src.QNetwork import QNetwork
from src.QDuelingNetwork import QDuelingNetwork
from src.QDuelingGraphNetwork import QDuelingGraphNetwork
from src.ExperienceSampler import ExperienceSampler
from src.ActionSelector import ActionSelector
from src.TrainLogger import TrainLogger
from src.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from src.BoltzmannPolicy import BoltzmannPolicy
from src.DQLearning import DQLearning
import gymnasium as gym
import torch
from pathlib import Path


class ConfigurationLoader(object):


    """
    ConfigurationLoader class reads the json file and prepares the
    training parameters.
    """

    def __init__(self, input_path: Path, output_path: Path):
        """Reads a json file and loads the parameters for setup
        the DRL agent training.

        Args:
            input_path (Path): The path of the file from where the
            configuration is loaded.
            output_dir (Path): The output directory for the model and
            the training data.
        """


        self.configuration = json.load(open(input_path))
        self.models_data_path: Path = Path.joinpath(output_path, "models")
        self.runs_data_path: Path = Path.joinpath(output_path, "runs")
        #Set the device for torch
        self.device = torch.device("cpu")
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        self.set_env()

    def set_env(self) -> gym.Env:
        """
        Sets the gym enviroment where the agent will be trained.
        """
        config = self.configuration["env"]
        env_name = config["name"]
        env_params = None
        self.env = gym.make(env_name)


    def get_optimizer(self,
                      optim_id, policy_net,
                      learning_rate) -> torch.optim.Optimizer:
        """
        Maps a configuration parameter to a PyTorch optimizer.
        """
        if (optim_id == "adamw"):
            optimizer = torch.optim.AdamW(policy_net.parameters(),
                                          lr=learning_rate,
                                          amsgrad=True)
        elif (optim_id == "adam"):
            optimizer = torch.optim.Adam(policy_net.parameters(),
                                         lr=learning_rate,
                                         amsgrad=True)
        else:
            optimizer = torch.optim.AdamW(policy_net.parameters(),
                                          lr=learning_rate,
                                          amsgrad=True)
        return optimizer

    def get_loss_fn(self, loss_fn_id):
        """
        Maps a configuration parameter to a PyTorch optimizer.
        """
        if (loss_fn_id == "mse"):
            loss_fn = torch.nn.MSELoss()
        elif (loss_fn_id == "mae"):
            loss_fn = torch.nn.L1Loss()
        elif (loss_fn_id == "crossentropy"):
            loss_fn = torch.nn.CrossEntropyLoss()
        elif (loss_fn_id == "huber"):
            loss_fn = torch.nn.HuberLoss()

        return loss_fn

    def get_q_estimator(self) -> QEstimator:
        """
        Given the configuration loaded, create and return a QEstimator.
        """
        n_obs = gym.spaces.utils.flatten_space(
                    self.env.observation_space).shape[0]
        n_act = gym.spaces.utils.flatten_space(self.env.action_space).shape[0]
        config = self.configuration["hyperparameters"]["q_estimator"]
        network_type = "dueling"
        if ("type" in config):
            network_type = config["type"]
        layers = config["layers"]
        if (network_type == "graph_dueling"):
            n_uavs = len(self.env.get_wrapper_attr("network_graph").uav_list)
            n_mss = len(self.env.get_wrapper_attr("network_graph").ms_list)
            policy_net = QDuelingGraphNetwork(n_obs,
                                              n_uavs,
                                              n_mss,
                                              layers,
                                              device = self.device)
        if (network_type == "dueling"):
            policy_net = QDuelingNetwork(n_obs,
                                         n_act,
                                         layers,
                                         device = self.device)
        elif (network_type == "simple"):
            policy_net = QNetwork(n_obs,
                                  n_act,
                                  layers,
                                  device = self.device)
        optim_id = config["optimizer"]
        learning_rate = config["learning_rate"]
        optim = self.get_optimizer(optim_id,
                                   policy_net,
                                   learning_rate)
        loss_fn_id = config["loss_fn"]
        loss_fn = self.get_loss_fn(loss_fn_id)
        update_policy = config["update_policy"]
        update_param = config["update_param"]
        gamma = config["gamma"]
        variation = self.configuration["variation"]
        target_net = None
        if (variation == "ddqn" or variation == "target"):
            if (network_type == "graph_dueling"):
                n_uavs = len(self.env.get_wrapper_attr("network_graph").uav_list)
                n_mss = len(self.env.get_wrapper_attr("network_graph").ms_list)
                target_net = QDuelingGraphNetwork(n_obs,
                                  n_uavs,
                                  n_mss,
                                  layers,
                                  device = self.device)
            if (network_type == "dueling"):
                target_net = QDuelingNetwork(n_obs,
                                             n_act,
                                             layers,
                                             device = self.device)
            elif (network_type == "simple"):
                target_net = QNetwork(n_obs,
                                      n_act,
                                      layers,
                                      device = self.device)
            target_net.load_state_dict(policy_net.state_dict())
        if (not Path.exists(self.models_data_path)):
            os.makedirs(self.models_data_path)
        output_path: Path = Path.joinpath(self.models_data_path,
                                          self.configuration["id"] + ".pt")
        q_estimator = QEstimator(policy_net,
                                 optim,
                                 loss_fn,
                                 gamma,
                                 self.device,
                                 update_policy,
                                 update_param,
                                 target_net,
                                 variation,
                                 output_path)
        return q_estimator

    def get_action_selector(self) -> ActionSelector:
        """
        Given the configuration, create and return an ActionSelector
        """
        config = self.configuration["hyperparameters"]["policy"]
        policy_type = config["type"]
        decay_strat = config["decay_strat"]
        start_expl_rate = config["start_expl_rate"]
        end_expl_rate = config["end_expl_rate"]
        decay_rate = config["decay_rate"]
        if (policy_type == "epsilon"):
            policy = EpsilonGreedyPolicy(start_expl_rate)
        elif (policy_type == "boltzmann"):
            policy = BoltzmannPolicy(start_expl_rate)
        actionSelector = ActionSelector(
            policy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_expl_rate,
            end_exploration_rate=end_expl_rate,
            decay_rate= decay_rate)

        return actionSelector

    def get_memory(self) -> ExperienceSampler:
        """
        Given the configuration, create and return an ExperienceSampler.
        """
        config = self.configuration["hyperparameters"]["memory"]
        memory_type = config["type"]
        memory_size = config["memory_size"]
        epsilon = config["epsilon"]
        alpha = config["alpha"]
        memory = ExperienceSampler(memory_size,
                                   memory_type,
                                   self.device,
                                   epsilon,
                                   alpha)
        return memory

    def get_parameters(self) -> dict:
        """
        Given the configuration, create a dictionary with the training
        parameters. Also setup the TrainLogger.
        """
        parameters = {}
        parameters["id"] = self.configuration["id"]
        parameters["env"] = self.env
        parameters["device"] = self.device

        parameters["logger"] = TrainLogger(self.runs_data_path,
                                           parameters["id"],
                                           json.dumps(self.configuration,
                                                      indent=4))
        config = self.configuration["hyperparameters"]
        parameters["training_steps"] = config["training_steps"]
        parameters["batches"] = config["batches"]
        parameters["batch_size"] = config["batch_size"]
        parameters["updates_per_batch"] = config["updates_per_batch"]
        parameters["h"] = config["h"]
        parameters["q_estimator"] = self.get_q_estimator()
        parameters["action_selector"] = self.get_action_selector()
        parameters["memory"] = self.get_memory()
        return parameters

    def get_agent(self):
        """
        Get the agent ready to train.
        """
        parameters = self.get_parameters()
        agent = DQLearning(parameters)
        return agent


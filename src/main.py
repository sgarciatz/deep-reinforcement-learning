import argparse
import random
import numpy as np
import torch
import time
from pathlib import Path
from src.ConfigurationLoader import ConfigurationLoader


def parse_arguments ():
    args_parser = argparse.ArgumentParser(
        description= ("Create DRL Agent to learn how "
                      "to perform in a given enviroment."))
    args_parser.add_argument("--input-path",
                             metavar="I",
                             type=Path,
                             nargs=1,
                             help=("The path of the input file "
                                   "with the agent parameters."))
    args_parser.add_argument("--output-path",
                             metavar="O",
                             type=Path,
                             nargs=1,
                             help=("The path of the output directory"))
    args_parser.add_argument("-t",
                             "--test",
                             nargs="?",
                             help="Test an already trained model",
                             type=bool,
                             const=True)
    return args_parser.parse_args()

def train(input_path: Path,
          output_path: Path):
    """Train an agent using the parameters specified in the input file.

    Args:
        input_path (Path): The path of the file that contains the
        parameters.
        output_path (Path): The path of the directory that will contain
        the info about the training and the resulting model.
    Returns:
        DQLearning: The Trained Agent.
    """
    agent = ConfigurationLoader(input_path=input_path,
                                output_path=output_path).get_agent()
    agent.train()
    return agent

def test_agent(input_file, agent = None):

    """Use a trained agent to test its performance. If the program is
    executed without the "--test" flag the model is initialized here.

    Parameters:
    - input_file = The path of the file containing the agent
      information.
    - agent = The initialized agent.
    """

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if (agent is None):
        agent = ConfigurationLoader(input_file).get_agent()
    agent.q_estimator.load_model()
    start_time = time.time()
    reward, ep_length, jumps = agent.test(100)
    print(f"Elapsed time = {time.time() - start_time }")
    print(reward, ep_length, jumps)


def main(args):

    input_path = args.input_path[0]
    output_path = args.output_path[0]
    is_testing = args.test
    if (is_testing is None):
        agent = train(input_path=input_path,
                      output_path=output_path)
    test_agent(input_path, agent)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main(parse_arguments())

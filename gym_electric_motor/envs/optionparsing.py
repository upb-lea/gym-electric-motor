import argparse
import json
import numpy as np
import os.path


def parse_args():
    """
    Option parsing function for the gym motor environments.

    **Options:**
        --env_id:               Environment id that is used for the gym.make(env_id) call.\

        --env_parameter:        Parameter json file path, that contains parameters that can be set in the environments
                                | constructor except load_parameter and motor_parameter.\

        --motor_parameter       Parameter json file path that contains the technical motor parameters.
                                | These parameters will be set at env_parameter['motor_parameter'] when returned.\

        --load_parameter        Parameter json file path that contains the technical load parameters.
                                | These parameters will be set at env_parameter['load_parameter'] when returned.\

        --training_parameter    Parameter json file path, that contains parameters that can be set for the agent.
                                | See example below.

        --load_weights


        --store_weights


        --load_model

        --store_model

    **Examples:**

        Training Parameter File:
            {
                  "memory_parameter"(dict) :        Parameter that can be set in the constructor of the SequentialMemory

                  "random_process"(str) :           Name of the Random Process (only used for DDPG),

                  "random_process_parameter"(dict): Parameter for the random process (only used for DDPG)

                  "policy"(str) :                   Name of the policy (only used for DQN)

                  "policy_parameter"(dict):         Parameter for the policy (only used for DQN)

                  "agent_parameter"(dict):          Parameter to set in the agents constructor

                  "optimizer"(str):                 Name of the Optimizer

                  "optimizer_parameter"(dict):      Parameter of the Optimizer

                  "metrics"(list(str)):             Metrics for the optimizer

                  "fit"(bool):                      Flag if the model shall be fit.

                  "fit_parameter"(dict):            Parameter for the fitting.

                  "test"(bool):                     Flag if the model shall be tested.

                  "test_parameter"(dict):           Parameter for testing.
            }
    }

    Returns:
        A tuple of the parsed (environment_id, environment_parameters, training_parameters, load_weights_path,
        store_weights_path, load_model_path, store_model_path)
    """
    arg_parser = argparse.ArgumentParser(description="Reinforcement Learning for DC Motors.")
    arg_parser.add_argument('--env_id', type=str, help='Identification id for the environment.')
    arg_parser.add_argument('--env_parameter', type=str, help='File path to the json containing (a subset)'
                                                              ' of the environment parameters. All parameters'
                                                              ' that are not in this file will be set to default')
    arg_parser.add_argument('--load_parameter', type=str, help='File path to the json containing (a subset)'
                                                               ' of the load parameters [a,b,c, J_load], '
                                                               'or the number of the standard parameters'
                                                               'All parameters that are not in this file will be'
                                                               ' set to default')
    arg_parser.add_argument('--motor_parameter', type=str, help='File path to the json containing (a subset)'
                                                                ' of the motor parameters or the number of the standard'
                                                                ' parameters All parameters'
                                                                ' that are not in this file will be set to default')
    arg_parser.add_argument('--training_parameter', type=str, help='File path to the json containing (a subset)'
                                                                   ' of the training parameters. All parameters'
                                                                   ' that are not in this file will be set to default')
    arg_parser.add_argument('--load_weights', type=str, help='Path to the model weights to load.'
                                                             'For agents with more than one neural network'
                                                             'have a look in its documentation for the order'
                                                             'of the paths')
    arg_parser.add_argument('--save_weights', type=str, help='Path to the model weights to store.'
                                                             'For agents with more than one neural network'
                                                             'have a look in its documentation for the order'
                                                             'of the paths')
    arg_parser.add_argument('--load_models', type=str, help='Path to the models to load.'
                                                            'For agents with more than one neural network'
                                                            'have a look in its documentation for the order'
                                                            'of the paths')
    arg_parser.add_argument('--save_models', type=str, help='Path to the models to store.'
                                                            'For agents with more than one neural network'
                                                            'have a look in its documentation for the order'
                                                            'of the paths')
    arg_parser.add_argument('--log_file', type=str, help='Path of the log file to be stored')
    arguments = arg_parser.parse_args()
    args_dict = arguments.__dict__

    # If an environment parameter file was passed take these parameters, otherwise an empty_dictionary
    if args_dict['env_parameter'] is not None:
        with open(args_dict['env_parameter']) as json_file:
            env_dict = json.load(json_file)
    else:
        env_dict = {}

    # If a load parameter file was passed take these parameters or the number
    if args_dict['load_parameter'] is not None:
        try:
            env_dict['load_parameter'] = int(args_dict['load_parameter'])
        except ValueError:
            with open(args_dict['load_parameter']) as json_file:
                env_dict['load_parameter'] = json.load(json_file)

    # If a motor parameter file was passed take these parameters or the number
    if args_dict['motor_parameter'] is not None:
        try:
            env_dict['motor_parameter'] = int(args_dict['motor_parameter'])
        except ValueError:
            with open(args_dict['motor_parameter']) as json_file:
                env_dict['motor_parameter'] = json.load(json_file)

    # Empty Training dict to use always the default training values.
    train_dict = {
        'memory_parameter': {'limit': 50000, 'window_length': 1, 'ignore_episode_boundaries': True},
        'policy': '',
        'policy_parameter': {},
        'random_process': '',
        'random_process_parameter': {},
        'agent_parameter': {},
        'optimizer': '',
        'optimizer_parameter': {},
        'metrics': [],
        'fit': True,
        'fit_parameter': {'nb_steps': 50000, 'visualize': True},
        'test': True,
        'test_parameter': {'nb_episodes': 20, 'visualize': True}
    }

    # If a load parameter file was passed, update the empty default dict
    if args_dict['training_parameter'] is not None:
        with open(args_dict['training_parameter']) as json_file:
            train_dict.update(json.load(json_file))

    return args_dict['env_id'], env_dict, train_dict, args_dict['load_weights'], args_dict['save_weights'],\
        args_dict['load_models'], args_dict['save_models'], args_dict['log_file']

from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import time

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Consists of Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(BasicLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class DQN(nn.Module):
    """
    DQN with Skip Connections: This model processes the input feature matrix and predicts Q-values for different actions.
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape  # Example: (7, 7, 6)
        self.num_actions = num_actions  # Number of actions

        # Defining convolutional layers
        self.conv1 = BasicLayer(6, 16, kernel_size=3, padding=1)
        self.conv2 = BasicLayer(16, 32, kernel_size=3, padding=1)
        self.conv3 = BasicLayer(32, 64, kernel_size=3, padding=1)

        # Skip connection: Process input directly to later layer
        self.skip = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1),  # 1x1 convolution for channel alignment
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Compute the output size after convolution layers
        self.conv_output_size = self._get_conv_out(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape).permute(0, 3, 1, 2))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        # Applying skip connection
        skip_out = self.skip(x)

        # Passing through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        # Combining with skip connection
        x = x + skip_out
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    num_actions = 6

    # setup device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if self.train and os.path.isfile("my-saved-model_mse_rms.pt"):
        with open("my-saved-model_mse_rms.pt", "rb") as file:
            self.policy_net = pickle.load(file).to(device)

    else:
        self.policy_net = DQN((7, 7, 6), num_actions).to(device)

    self.target_net = DQN((7, 7, 6), num_actions).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3, centered=True)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # TODO: 更好的 memory 利用对称性


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

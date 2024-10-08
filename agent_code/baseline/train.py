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
from tqdm import tqdm

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 50000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
F_TRAIN = 100
NUM_EPOCH = 10
BATCH_SIZE = 256
GAMMA = 0.5
TAU = 0.05
LR = 1e-3

DO_PLOTS = True

def update_plots(scores):
    plt.close('all')
    plt.figure(figsize=(12, 5))

    plt.plot(scores)
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.title('Scores Over Round')
    
    plt.tight_layout()
    plt.savefig('./plots/result.png')

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Consists of Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
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
        self.conv1 = BasicLayer(5, 8)
        self.conv2 = BasicLayer(8, 16)
        self.conv3 = BasicLayer(16, 32)
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
        # Passing through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

    if self.train and os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file).to(device)

    else:
        self.policy_net = DQN((17, 17, 5), num_actions).to(device)

    self.target_net = DQN((17, 17, 5), num_actions).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.num_round = 0
    self.history_score = []


# Action rotation mapping (for 0, 90, 180, 270 degrees)
ROTATION_MAPPING = {
    0: {'UP': 'UP', 'RIGHT': 'RIGHT', 'DOWN': 'DOWN', 'LEFT': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    90: {'UP': 'LEFT', 'RIGHT': 'UP', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    180: {'UP': 'DOWN', 'RIGHT': 'LEFT', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    270: {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
}

# Action flipping mapping (for horizontal and vertical flips)
VERTICAL_FLIP_MAPPING = {'UP': 'DOWN', 'RIGHT': 'RIGHT', 'DOWN': 'UP', 'LEFT': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
HORIZONTAL_FLIP_MAPPING = {'UP': 'UP', 'RIGHT': 'LEFT', 'DOWN': 'DOWN', 'LEFT': 'RIGHT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}

# Function to adjust action based on rotation and flip
def adjust_action(action, rotation=0, flip_horizontal=False, flip_vertical=False):
    # Adjust action based on rotation
    action = ROTATION_MAPPING[rotation][action]
    if flip_horizontal:
        action = HORIZONTAL_FLIP_MAPPING[action]
    if flip_vertical:
        action = VERTICAL_FLIP_MAPPING[action]

    return action

# Function to adjust the state (rotate and flip) while considering C x H x W format
def adjust_state(state, rotation=0, flip_horizontal=False, flip_vertical=False):
    if rotation != 0:
        state = np.rot90(state, k=rotation // 90, axes=(1, 2))
    if flip_horizontal:
        state = np.flip(state, axis=2)
    if flip_vertical:
        state = np.flip(state, axis=1)

    return state


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

    # # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # Precompute features to avoid repetitive state_to_features calls
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    # Define all transformation combinations (rotation, horizontal flip, vertical flip)
    rotation_angles = [0, 90, 180, 270]
    flip_combinations = [(False, False), (True, False), (False, True)]  # (horizontal_flip, vertical_flip)

    # Iterate through each rotation angle
    for rotation in rotation_angles:
        # Iterate through each flip combination
        for flip_horizontal, flip_vertical in flip_combinations:
            # Adjust the state based on rotation and flip
            adjusted_old_state = adjust_state(old_features, rotation, flip_horizontal, flip_vertical)
            adjusted_new_state = adjust_state(new_features, rotation, flip_horizontal, flip_vertical)
            
            # Adjust the action based on rotation and flip
            adjusted_action = adjust_action(self_action, rotation, flip_horizontal, flip_vertical)

            # Append the augmented transition to the queue
            self.transitions.append(Transition(
                torch.Tensor(adjusted_old_state.copy()).type('torch.FloatTensor').to(device),
                torch.Tensor([[ACTIONS.index(adjusted_action)]]).to(torch.int64).to(device),
                torch.Tensor(adjusted_new_state.copy()).type('torch.FloatTensor').to(device),
                torch.Tensor([reward]).type('torch.FloatTensor').to(device)
            ))


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
    self.num_round += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(
        torch.Tensor(state_to_features(last_game_state)).type('torch.FloatTensor').to(device), 
        torch.Tensor([[ACTIONS.index(last_action)]]).to(torch.int64).to(device), 
        torch.zeros_like(torch.Tensor(state_to_features(last_game_state))).type('torch.FloatTensor').to(device), 
        torch.Tensor([reward_from_events(self, events)]).type('torch.FloatTensor').to(device)
        ))
    # Store game score for analysis
    self.history_score.append(last_game_state['self'][1])
    # Only train the agent every N rounds
    if self.num_round % F_TRAIN == 0:  # N is the number of rounds between training sessions
        for epoch in tqdm(range(NUM_EPOCH)):  # Perform NUM_EPOCH training iterations

            # Shuffle the entire transitions buffer
            transition_list = list(self.transitions)
            random.shuffle(transition_list)

            # Split the transitions into batches and train on each batch
            for i in range(0, len(transition_list), BATCH_SIZE):
                batch_data = Transition(*zip(*transition_list[i:i + BATCH_SIZE]))
                action_batch = torch.cat(batch_data.action)
                reward_batch = torch.cat(batch_data.reward)
                state_batch = torch.stack(batch_data.state)
                next_state_batch = torch.stack(batch_data.next_state)

                state_values = self.policy_net(state_batch)
                state_values = state_values.gather(1, action_batch)
                with torch.no_grad():
                    next_state_values = self.target_net(next_state_batch)
                next_state_values = next_state_values.max(1)[0]
                expected_state_values = (next_state_values * GAMMA) + reward_batch

                criterion = nn.MSELoss()
                loss = criterion(state_values, expected_state_values.unsqueeze(1))

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                self.optimizer.step()

        # Update the target network after NUM_EPOCH training iterations
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        update_plots(self.history_score)
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: 0,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0.3,
        e.CRATE_DESTROYED: 1.,
        e.COIN_FOUND: 0.5,
        e.COIN_COLLECTED: 2.,
        e.KILLED_OPPONENT: 20.,
        e.KILLED_SELF: -50.,
        e.GOT_KILLED: -10.,
        e.SURVIVED_ROUND: 5.
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

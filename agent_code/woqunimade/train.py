from collections import namedtuple, deque

import os
import json
from typing import List

import events as e
from .callbacks import state_to_features, feat2str, ACTIONS

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
MOVE_TO_TARGET = "MOVE_TO_TARGET"
MOVE_TO_DANGER = "MOVE_TO_DANGER"
TRY_TO_ESCAPE = "TRY_TO_ESCAPE"
ATTACK_CRATE = "ATTACK_CRATE"
ATTACK_ENEMY = "ATTACK_ENEMY"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.epsilon = 0.8
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.9995
    self.gamma = 0.65
    self.alpha = 0.05
    if os.path.isfile("q_tabular.json"):
        with open("q_tabular.json", "r") as file:
            self.q_tabular = json.load(file)
    else:
        self.q_tabular = dict()


ROTATION_MAPPING = {
    0: {'UP': 'UP', 'RIGHT': 'RIGHT', 'DOWN': 'DOWN', 'LEFT': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    90: {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    180: {'UP': 'DOWN', 'RIGHT': 'LEFT', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    270: {'UP': 'LEFT', 'RIGHT': 'UP', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
}

# Action flipping mapping (for horizontal and vertical flips)
VERTICAL_FLIP_MAPPING = {'UP': 'DOWN', 'RIGHT': 'RIGHT', 'DOWN': 'UP', 'LEFT': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
HORIZONTAL_FLIP_MAPPING = {'UP': 'UP', 'RIGHT': 'LEFT', 'DOWN': 'DOWN', 'LEFT': 'RIGHT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}


def adjust_action(action, rotation=0, flip_horizontal=False, flip_vertical=False):
    # Adjust action based on rotation
    action = ROTATION_MAPPING[rotation][action]
    if flip_horizontal:
        action = HORIZONTAL_FLIP_MAPPING[action]
    if flip_vertical:
        action = VERTICAL_FLIP_MAPPING[action]

    return action


def adjust_state(state, rotation=0, flip_horizontal=False, flip_vertical=False):
    direction_index = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # Define the initial order of directions before rotation

    # Normalize and apply clockwise rotation
    rotation = (rotation // 90) % 4  # Normalize the rotation to one of 0, 1, 2, or 3
    directions = directions[-rotation:] + directions[:-rotation]  # Rotate the list using slicing

    # Handle horizontal flip
    if flip_horizontal:
        directions[1], directions[3] = directions[3], directions[1]  # Swap 'RIGHT' and 'LEFT'

    # Handle vertical flip
    if flip_vertical:
        directions[0], directions[2] = directions[2], directions[0]  # Swap 'UP' and 'DOWN'

    # Update the state according to the new direction list
    new_state = [0] * 6
    for i, dir in enumerate(directions):
        new_state[i] = state[direction_index[dir]]
    new_state[4] = state[4]  # 'CENTER' remains unchanged
    new_state[5] = state[5]  # 'BOMBS_LEFT' remains unchanged

    return new_state


def update_table(self, old_feature: List[str], self_action: str, new_feature: List[str], reward: float):
    old_feature_str = feat2str(self, old_feature)
    new_feature_str = feat2str(self, new_feature)

    action_idx = ACTIONS.index(self_action)
    Q_s0_a0 = self.q_tabular[old_feature_str][action_idx]
    delta = reward + self.gamma * max(self.q_tabular[new_feature_str]) - Q_s0_a0
    Q_s1_a1 = Q_s0_a0 + self.alpha * delta
    self.q_tabular[old_feature_str][action_idx] = Q_s1_a1



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

    old_feature = state_to_features(old_game_state)
    new_feature = state_to_features(new_game_state)
    # Idea: Add your own events to hand out rewards
    if old_feature[ACTIONS.index(self_action)] == "target":
        events.append(MOVE_TO_TARGET)
    if old_feature[ACTIONS.index(self_action)] == "danger":
        events.append(MOVE_TO_DANGER)
    if old_feature[ACTIONS.index(self_action)] == "escape":
        events.append(TRY_TO_ESCAPE)
    if self_action == 'BOMB':
        if old_feature[-2]:
            events.append(ATTACK_CRATE)
        elif old_feature[-1]:
            events.append(ATTACK_ENEMY)

    reward = reward_from_events(self, events)

    # Define all transformation combinations (rotation, horizontal flip, vertical flip)
    rotation_angles = [0, 90, 180, 270]
    flip_combinations = [(False, False), (True, False), (False, True)]  # (horizontal_flip, vertical_flip)

    # Iterate through each rotation angle
    for rotation in rotation_angles:
        # Iterate through each flip combination
        for flip_horizontal, flip_vertical in flip_combinations:
            # Adjust the state based on rotation and flip
            adjusted_old_feature = adjust_state(old_feature, rotation, flip_horizontal, flip_vertical)
            adjusted_new_feature = adjust_state(new_feature, rotation, flip_horizontal, flip_vertical)
            # Adjust the action based on rotation and flip
            adjusted_action = adjust_action(self_action, rotation, flip_horizontal, flip_vertical)
            update_table(self, adjusted_old_feature, adjusted_action, adjusted_new_feature, reward)
    


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

    reward = reward_from_events(self, events)
    last_feature_str = feat2str(self, state_to_features(last_game_state))

    action_idx = ACTIONS.index(last_action)
    Q_s0_a0 = self.q_tabular[last_feature_str][action_idx]
    delta = reward if 'GOT_KILLED' in events else 0 # some action will lead to death
    Q_s1_a1 = Q_s0_a0 + self.alpha * delta
    self.q_tabular[last_feature_str][action_idx] = Q_s1_a1

    if self.epsilon >= self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    # Store the model
    if last_game_state['round'] % 100 == 0:
        with open("q_tabular.json", "w") as file:
            file.write(json.dumps(self.q_tabular))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -5,
        e.BOMB_DROPPED: -10,
        e.INVALID_ACTION: -50,
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 100.,
        e.KILLED_SELF: -300.,
        e.GOT_KILLED: -100.,
        MOVE_TO_TARGET: 50,
        MOVE_TO_DANGER: -60,
        TRY_TO_ESCAPE: 60,
        ATTACK_CRATE: 40,
        ATTACK_ENEMY: 80,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

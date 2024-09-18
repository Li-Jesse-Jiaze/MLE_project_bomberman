from typing import List
import random

import events as e
from .callbacks import feat2str, choose_action, ACTIONS
from .table import Table
from .symmetry import adjust_action, adjust_state

# Events
MOVE_TO_DEAD = "MOVE_TO_DEAD"
MOVE_TO_TARGET = "MOVE_TO_TARGET"
ATTACK_TARGET = "ATTACK_TARGET"
ATTACK_ENEMY = "ATTACK_ENEMY"
KILL_ENEMY = "KILL_ENEMY"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.epsilon = 0.5
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.999
    self.gamma = 0.9
    self.alpha = 0.1
    self.q_table_1 = Table()
    self.q_table_2 = Table()
    try:
        self.q_table_1.load_from_json('q_table_1.json')
        self.q_table_2.load_from_json('q_table_2.json')
        self.logger.debug('Load table from file. Continue training.')
    except FileNotFoundError:
        self.logger.debug('Start training from an empty table.')


def update_table(self, q_1: Table, q_2: Table, old_feature: List[str], self_action: str, new_feature: List[str], reward: float):
    # Use q_2 update q_1
    s0 = feat2str(old_feature)
    a0 = ACTIONS.index(self_action)
    Q_1_s0a0 = q_1[s0][a0]
    if len(new_feature):
        s1 = feat2str(new_feature)
        Q_2_s1a = max(q_2[s1])
        delta = reward + self.gamma * Q_2_s1a - Q_1_s0a0
    else: # End of the round
        delta = reward - Q_1_s0a0
    q_1[s0][a0] = Q_1_s0a0 + self.alpha * delta


def learn(self, old_feature: List[str], self_action: str, new_feature: List[str], reward: float):
    if random.random() < 0.5:
        update_table(self, self.q_table_1, self.q_table_1, old_feature, self_action, new_feature, reward)
    else:
        update_table(self, self.q_table_2, self.q_table_1, old_feature, self_action, new_feature, reward)
        

def learn_symmetrically(self, old_feature: List[str], self_action: str, new_feature: List[str], reward: float):
    rotation_angles = [0, 90, 180, 270]
    flip_combinations = [(False, False), (True, False), (False, True)]
    for rotation in rotation_angles:
        for flip_horizontal, flip_vertical in flip_combinations:
            # Adjust state
            adjusted_old_feature = adjust_state(old_feature, rotation, flip_horizontal, flip_vertical)
            adjusted_new_feature = adjust_state(new_feature, rotation, flip_horizontal, flip_vertical)
            # Adjust action
            adjusted_action = adjust_action(self_action, rotation, flip_horizontal, flip_vertical)
            learn(self, adjusted_old_feature, adjusted_action, adjusted_new_feature, reward)


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

    old_feature = self.feature(old_game_state)
    new_feature = self.feature(new_game_state)
    # Idea: Add your own events to hand out rewards
    if old_feature[ACTIONS.index(self_action)] == "target":
        events.append(MOVE_TO_TARGET)
    if old_feature[ACTIONS.index(self_action)] == "dead":
        events.append(MOVE_TO_DEAD)
    if self_action == 'BOMB':
        if 'enemy' in old_feature:
            events.append(ATTACK_ENEMY)
        if old_feature[-1] == 'target':
            events.append(ATTACK_TARGET)
        if old_feature[-1] == 'KILL!':
            events.append(KILL_ENEMY)

    reward = reward_from_events(self, events)

    learn_symmetrically(self, old_feature, self_action, new_feature, reward)
    


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

    old_feature = self.feature(last_game_state)
    new_feature = []
    reward = reward_from_events(self, events)

    learn_symmetrically(self, old_feature, last_action, new_feature, reward)

    if self.epsilon >= self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    # Store the model
    if last_game_state['round'] % 100 == 0:
        self.q_table_1.save_to_json("q_table_1.json")
        self.q_table_2.save_to_json("q_table_2.json")


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
        e.BOMB_DROPPED: -5,
        e.INVALID_ACTION: -20,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -300,
        e.GOT_KILLED: -100.,
        MOVE_TO_DEAD: -100,
        MOVE_TO_TARGET: 50,
        ATTACK_TARGET: 50,
        ATTACK_ENEMY: 20,
        KILL_ENEMY: 500,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

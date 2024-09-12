from .table import Table
from .features import Feature
import random
from typing import List

import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def feat2str(feature):
    return ", ".join(feature)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not self.train:
        self.q_table = Table()
        self.q_table.load_from_json('q_table.json')
        self.logger.info("Loading model from saved state.")
    self.feature = Feature()


def choose_action(self, feature: List[str]) -> str:
    feature_str = feat2str(feature)
    # Find best action
    Q = np.array(self.q_table[feature_str])
    best_choice = np.where(Q == max(Q))[0]
    best_action = ACTIONS[np.random.choice(best_choice)]
    # todo Exploration vs exploitation
    if self.train:
        if random.random() < self.epsilon:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        else:
            self.logger.debug(f"Querying model for action.:{best_action}")
            return best_action
    else:
        self.logger.debug(f"Querying model for action.:{best_action}")
        return best_action



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = self.feature(game_state)
    self.logger.debug(f"feature: {feat2str(features)}")

    # If train with itself
    if game_state["round"] % 500 == 0 and game_state["step"] == 1:
        setup(self)

    return choose_action(self, features)

import os
import pickle
import random

import numpy as np
import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    f_state = torch.tensor(state_to_features(game_state), device=device, dtype=torch.float).unsqueeze(0)
    random_prob = .1
    if self.train:
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            self.logger.debug("Choosing action based on policy_net.")
            return ACTIONS[torch.multinomial(self.policy_net(f_state)[0], 1)[0]]
    
    p_action = self.model(f_state)[0]
    i = 1
    while True:
        action_idx = int(torch.topk(p_action.flatten(), i).indices[-1])
        if check_inv_action(game_state, ACTIONS[action_idx]):
            break
        i += 1

    self.logger.debug("Querying model for action.")
    return ACTIONS[action_idx]


def check_inv_action(game_state: dict, action: str) -> bool:
    field = game_state["field"]
    my_pos = game_state["self"][3]

    # Map each action to its corresponding position adjustment
    move_offsets = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0)
    }

    if action in move_offsets:
        dx, dy = move_offsets[action]
        new_pos = (my_pos[0] + dx, my_pos[1] + dy)
        # Check if the new position is blocked
        if field[new_pos[0], new_pos[1]] != 0:
            return False
    return True

def state_to_features(game_state: dict, reach: int = 3) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    agent = game_state["self"]
    others = [o[3] for o in game_state["others"]]  # Extract other agents' positions

    my_pos = np.array(agent[3])

    # Generate a matrix of positions around the player
    vision_range = np.arange(-reach, reach + 1)
    grid_x, grid_y = np.meshgrid(vision_range, vision_range, indexing='ij')
    vision_coords = np.stack([grid_x, grid_y], axis=-1) + my_pos

    # Clipping coordinates to ensure they are within field boundaries
    vision_coords_clipped = np.clip(vision_coords, [0, 0], np.array(field.shape) - 1)

    # Define feature layers
    feat_walls = (field[vision_coords_clipped[:, :, 0], vision_coords_clipped[:, :, 1]] == -1).astype(int)
    feat_crates = (field[vision_coords_clipped[:, :, 0], vision_coords_clipped[:, :, 1]] == 1).astype(int)
    feat_explosions = (explosion_map[vision_coords_clipped[:, :, 0], vision_coords_clipped[:, :, 1]] > 0).astype(int)
    feat_coins = np.zeros_like(feat_walls)
    feat_bombs = np.zeros_like(feat_walls)
    feat_opponents = np.zeros_like(feat_walls)

    # Mark coins, bombs, and opponents in the respective feature layers
    for coin in coins:
        coin_pos = np.array(coin) - my_pos + reach
        if 0 <= coin_pos[0] < 2 * reach + 1 and 0 <= coin_pos[1] < 2 * reach + 1:
            feat_coins[coin_pos[0], coin_pos[1]] = 1

    for bomb in bombs:
        bomb_pos = np.array(bomb[0]) - my_pos + reach
        if 0 <= bomb_pos[0] < 2 * reach + 1 and 0 <= bomb_pos[1] < 2 * reach + 1:
            feat_bombs[bomb_pos[0], bomb_pos[1]] = 1

    for other in others:
        other_pos = np.array(other) - my_pos + reach
        if 0 <= other_pos[0] < 2 * reach + 1 and 0 <= other_pos[1] < 2 * reach + 1:
            feat_opponents[other_pos[0], other_pos[1]] = 1

    # Stack all layers into a single numpy array
    features = np.stack([feat_walls, feat_crates, feat_explosions, feat_coins, feat_bombs, feat_opponents])

    return features 
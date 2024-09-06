import os
import pickle
import random

import numpy as np
np.set_printoptions(linewidth=np.inf)
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
    random_prob = .2
    if self.train:
        if random.random() < random_prob:
            self.logger.debug("Choosing action based on Max-Boltzmann strategy.")
            logits = self.policy_net(f_state)[0]
            probabilities = torch.softmax(logits, dim=0).cpu().detach().numpy()  # Softmax to convert scores to probabilities
            return np.random.choice(ACTIONS, p=probabilities)
        else:
            self.logger.debug("Choosing action based on policy_net.")
            return ACTIONS[torch.argmax(self.policy_net(f_state)[0])]
    
    p_action = self.model(f_state)[0]
    i = 1
    while True:
        action_idx = int(torch.topk(p_action.flatten(), i).indices[-1])
        if check_inv_action(game_state, ACTIONS[action_idx]):
            break
        i += 1

    self.logger.debug(f"Querying model for action {ACTIONS[action_idx]}.")
    return ACTIONS[action_idx]


def check_inv_action(game_state: dict, action: str) -> bool:
    # Gather information about the game state
    arena = game_state["field"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state["others"]]  # noqa: F811

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if (
            (arena[d] == 0)
            and (game_state["explosion_map"][d] < 1)
            and (d not in others)
            and (d not in bomb_xys)
        ):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles:
        valid_actions.append("LEFT")
    if (x + 1, y) in valid_tiles:
        valid_actions.append("RIGHT")
    if (x, y - 1) in valid_tiles:
        valid_actions.append("UP")
    if (x, y + 1) in valid_tiles:
        valid_actions.append("DOWN")
    if (x, y) in valid_tiles:
        valid_actions.append("WAIT")
    if (bombs_left > 0):
        valid_actions.append("BOMB")
    
    return action in valid_actions

def state_to_features(game_state: dict, max_size=17) -> np.array:
    # Gather information about the game state
    arena = game_state["field"]
    my_pos = game_state["self"][-1]
    bombs = game_state["bombs"]
    coins = game_state["coins"]
    others = game_state["others"]  # Extract other agents' positions
    # Define feature layers
    # walls: -1, free: 0, crates: 1
    padding_size = max(0, max_size - arena.shape[0])
    if padding_size:
        f_arena = np.pad(arena,pad_width=((0, padding_size), (0, padding_size)),
                            mode='constant', constant_values=-1)
    else:
        f_arena = arena
    bomb_map = np.zeros(arena.shape)
    f_coins = np.zeros(arena.shape)
    f_others = np.zeros(arena.shape)
    f_agent = np.zeros(arena.shape)
    for (xb, yb), t in bombs:
        bomb_map[xb, yb] = max(bomb_map[xb, yb], 4 - t)
        # Expand explosion in four directions, stop if a wall is encountered
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Directions: left, right, up, down
            for h in range(1, 4):
                xi, yj = xb + h * direction[0], yb + h * direction[1]
                if not (0 <= xi < bomb_map.shape[0] and 0 <= yj < bomb_map.shape[1]) or arena[xi, yj] == -1:
                    break
                bomb_map[xi, yj] = max(bomb_map[xi, yj], 4 - t)
    f_bombs = bomb_map / 2.0 - 1
    for coin in coins:
        f_coins[coin[0], coin[1]] = 1
    for _, _, _, pos in others:
        f_others[pos[0], pos[1]] = 1
    f_agent[my_pos[0], my_pos[1]] = 1
    features = np.stack([f_arena, f_bombs, f_coins, f_others, f_agent])

    return features 

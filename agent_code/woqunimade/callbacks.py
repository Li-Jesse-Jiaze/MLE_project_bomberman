import os
import json
import random

import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def feat2str(self, feature):
    feature_str = ", ".join(feature)
    try:
        self.q_tabular[feature_str]
    except KeyError:
        self.q_tabular[feature_str] = list(np.zeros(6))
    return feature_str


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
    if self.train or not os.path.isfile("q_tabular.json"):
        self.logger.info("Setting up model from scratch.")
        self.q_tabular = dict()
    else:
        self.logger.info("Loading model from saved state.")
        with open("q_tabular.json", "r") as file:
            self.q_tabular = json.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    feature_str = feat2str(self, state_to_features(game_state))
    self.logger.debug(f"feature: {feature_str}")
    if self.train:
        if random.random() < self.epsilon:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        else:
            return ACTIONS[np.argmax(self.q_tabular[feature_str])]
    else:
        # if train with itself
        if game_state["round"] % 500 == 0 and game_state["step"] == 1:
            setup(self)
        try:
            Q = np.array(self.q_tabular[feature_str])
            best_choice = np.where(Q == max(Q))[0]
            self.logger.debug(f"Querying model for action.:{ACTIONS[np.random.choice(best_choice)]}")
            return ACTIONS[np.random.choice(best_choice)]
        except KeyError: # in case state not in table
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])


def look_for_targets(free_space, start, targets):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [
            (x, y)
            for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if free_space[x, y]
        ]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def danger_map(game_state: dict) -> np.array:
    field = game_state["field"]
    bombs = game_state["bombs"]
    danger_map = np.zeros(field.shape)
    for (xb, yb), t in bombs:
        danger_map[xb, yb] = max(danger_map[xb, yb], 4 - t)
        # Expand explosion in four directions, stop if a wall is encountered
        for direction in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]:  # Directions: left, right, up, down
            for h in range(1, 4):
                xi, yj = xb + h * direction[0], yb + h * direction[1]
                if (
                    not (
                        0 <= xi < danger_map.shape[0] and 0 <= yj < danger_map.shape[1]
                    )
                    or field[xi, yj] == -1
                ):
                    break
                danger_map[xi, yj] = max(danger_map[xi, yj], 4 - t)
    return danger_map


def state_to_features(game_state: dict):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: state in ['UP', 'RIGHT', 'DOWN', 'LEFT', 'CENTER', 'BOMBS_LEFT']
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    coins = game_state['coins']
    x, y = game_state['self'][-1]  # agent's current position

    # List to hold the surroundings data corresponding to the directions order
    # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'CENTER']
    features = []

    # Possible movements and their effects
    directions = [
        (x, y - 1),   # UP
        (x + 1, y),   # RIGHT
        (x, y + 1),   # DOWN
        (x - 1, y),   # LEFT
        (x, y)        # CENTER
    ]

    # Translate field values to strings
    field_dict = {
        1: 'crate',
        -1: 'wall',
        0: 'free'
    }
    
    danger = danger_map(game_state)
    free_space = field == 0
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)
    dead_ends = [
        (x, y)
        for x in cols
        for y in rows
        if (field[x, y] == 0)
        and (
            [field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(
                0
            )
            == 1
        )
    ]
    crates = [(x, y) for x in cols for y in rows if (field[x, y] == 1)]
    targets = coins + dead_ends + crates
    target = look_for_targets(free_space, (x, y), targets)
    # TODO: find target that really helps
    # TODO: find target for best escape if in dannger

    # Check surroundings
    for (i, j) in directions:
        tile = field_dict[field[i, j]]
        if any(bomb[0] == i and bomb[1] == j for bomb, _ in bombs):
            tile = 'bomb'
        elif any(coin[0] == i and coin[1] == j for coin in coins):
            tile = 'coin'
        elif any(opponent[-1] == (i, j) for opponent in game_state['others']):
            tile = 'enemy'
        elif tile == 'free':
            if danger[i, j] > 0:
                tile = 'danger'
            if (i, j) == target:
                tile = 'target'
            if explosion_map[i, j] > 0:
                tile = 'wall'

        features.append(tile)

    # TODO: record bomb history bombed before no more need
    features.append(str(game_state['self'][2]))

    return features

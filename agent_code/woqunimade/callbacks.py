import os
import json
import random
from collections import deque

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


def calculate_steps(field, pos, objects):
    distances = np.full((len(objects),), np.inf)
    target_index_map = {tuple(obj): idx for idx, obj in enumerate(objects)}
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    q = deque()
    q.append((pos[0], pos[1], 0))

    visited = set()
    visited.add(tuple(pos))
    
    # BFS
    while q:
        x, y, steps = q.popleft()
        if (x, y) in target_index_map:
            idx = target_index_map[(x, y)]
            distances[idx] = steps
            del target_index_map[(x, y)]
            if not target_index_map:
                break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, steps + 1))
    
    return distances


def look_for_target(game_state, features):
    field, _, _, coins, others, (x, y) = (
        game_state['field'], game_state['explosion_map'], game_state['bombs'],
        game_state['coins'], game_state['others'], game_state['self'][-1]
    )
    candidates = np.array(features)[:4]
    candidates = np.where(candidates == 'free')[0]
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    weights = {"crates":1, "coins":30, "enemy":1}

    if not len(candidates):
        return

    # Calculate scores, as described above:
    scores, best_indices = np.zeros(len(candidates)), []
    for i, index in enumerate(candidates):
        temp_pos = np.array(directions[index])
        # curr_tile = np.array([self_coordinates[0] + col, self_coordinates[1] + row])
        for object_name in weights.keys():
            if object_name == "crates":
                objects = np.array(np.where(field == 1))
                objects = objects.transpose()

            elif object_name == "coins":
                objects = np.array(coins)
            else:
                objects = np.array([op[-1] for op in others])

            if len(objects):
                distance = calculate_steps(field, temp_pos, objects)
                if len(distance):
                    temp = np.ones(len(distance)) * weights[object_name]
                    scores[i] += np.sum(temp / (distance + 1))


    best_indices = candidates[np.where(np.array(scores) == max(scores))[0]]

    j = np.random.choice(best_indices)
    features[j] = "target"


def find_safe_positions(pos, threshold, danger, field, explosion_map):
    x, y = pos
    max_rows = len(danger)
    max_cols = len(danger[0])
    safe_positions = []

    for i in range(max_rows):
        for j in range(max_cols):
            if abs(i - x) + abs(j - y) <= threshold and danger[i][j] == 0 and field[i][j] == 0 and explosion_map[i][j] == 0:
                safe_positions.append((i, j))
    
    return np.array(safe_positions)


def look_for_escape(game_state, features, danger):
    field, explosion_map, _, _, _, (x, y) = (
        game_state['field'], game_state['explosion_map'], game_state['bombs'],
        game_state['coins'], game_state['others'], game_state['self'][-1]
    )
    candidates = np.array(features)[:4]
    candidates = np.where(candidates == 'danger')[0]
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]

    if not len(candidates):
        return

    # Calculate the less step to free
    min_distance, best_indices = np.zeros(len(candidates)) * 5, []
    for i, index in enumerate(candidates):
        temp_pos = np.array(directions[index])
        objects = find_safe_positions(temp_pos, 4, danger, field, explosion_map)
        if len(objects):
            distance = calculate_steps(field, temp_pos, objects)
            if len(distance):
                min_distance[i] = distance.min()


    best_indices = candidates[np.where(np.array(min_distance) == min(min_distance))[0]]

    j = np.random.choice(best_indices)
    features[j] = "escape"


def danger_map(field: np.array, bombs: list) -> np.array:
    danger = np.zeros(field.shape)
    
    for (x_bomb, y_bomb), timer in bombs:
        update_danger(danger, field, x_bomb, y_bomb, timer)
        
    return danger


def update_danger(danger, field, x_bomb, y_bomb, timer):
    max_timer_value = 4
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down
    
    # Set danger level at bomb's location
    danger[x_bomb, y_bomb] = max(danger[x_bomb, y_bomb], max_timer_value - timer)
    
    # Expand danger level to adjacent squares unless blocked by a wall
    for dx, dy in directions:
        for distance in range(1, max_timer_value):  # bomb explosion range
            x, y = x_bomb + dx * distance, y_bomb + dy * distance
            if not (0 <= x < danger.shape[0] and 0 <= y < danger.shape[1]) or field[x, y] == -1:
                break
            danger[x, y] = max(danger[x, y], max_timer_value - timer)


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
    
    field, explosion_map, bombs, coins, others, (x, y) = (
        game_state['field'], game_state['explosion_map'], game_state['bombs'],
        game_state['coins'], game_state['others'], game_state['self'][-1]
    )

    features = []
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]  # UP, RIGHT, DOWN, LEFT, CENTER

    danger = danger_map(field, bombs)
    field_labels = {1: 'crate', -1: 'wall', 0: 'free'}
    
    for i, j in directions:
        tile = field_labels[field[i, j]]
        if tile == 'free':
            if any((b[0] == i and b[1] == j) for b, _ in bombs):
                tile = 'bomb'
            elif any((coin[0] == i and coin[1] == j) for coin in coins):
                tile = 'coin'
            elif any(op[-1] == (i, j) for op in others):
                tile = 'enemy'
            elif explosion_map[i, j] > 0:
                tile = 'explosion'
            elif danger[i, j] > 0:
                tile = 'danger'
        features.append(tile)
    if 'free' in features:
        look_for_target(game_state, features)
    elif 'danger' in features:
        look_for_escape(game_state, features, danger)
    features.append(str(game_state['self'][2]))  # Append bomb_left

    return features

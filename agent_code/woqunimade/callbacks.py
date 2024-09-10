import os
import json
import random
import copy
from collections import deque
import settings as s

import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # UP, RIGHT, DOWN, LEFT
DIRECTIONS_INCLUDING_WAIT = DIRECTIONS + [(0, 0)]


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


def calculate_steps(game_state, pos, objects, danger):
    field, explosion_map, bombs, _, others, (x, y) = (
        game_state['field'], game_state['explosion_map'], game_state['bombs'],
        game_state['coins'], game_state['others'], game_state['self'][-1]
    )
    distances = np.full((len(objects),), np.inf)
    target_index_map = {tuple(obj): idx for idx, obj in enumerate(objects)}
    directions = DIRECTIONS_INCLUDING_WAIT

    q = deque()
    q.append((pos[0], pos[1], 0))

    visited = set()
    visited.add(tuple(pos))
    
    # BFS
    while q:
        x, y, steps = q.popleft()
        if (danger[x][y] > 0 and s.BOMB_TIMER - danger[x][y] <= steps < s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER) \
            or explosion_map[x][y] > steps: # explosion while agent passing by
            continue
        if (x, y) in target_index_map:
            idx = target_index_map[(x, y)]
            distances[idx] = steps
            del target_index_map[(x, y)]
            if not target_index_map:
                break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and (nx, ny) not in visited:
                if field[nx, ny] != 0:
                    continue
                if any((nx, ny) == other[-1] for other in others) and steps == 0:
                    continue
                bomb_prevent = any((nx, ny) == (bx, by) and timer >= steps for (bx, by), timer in bombs)
                if bomb_prevent:
                    continue
                visited.add((nx, ny))
                q.append((nx, ny, steps + 1))
    
    return distances


def predict_next_state(game_state, bomb_drop=False):
    next_state = copy.deepcopy(game_state)
    
    field, explosion_map, bombs, _, _, (x, y) = (
        next_state['field'], next_state['explosion_map'], next_state['bombs'],
        next_state['coins'], next_state['others'], next_state['self'][-1]
    )
    new_bombs = []
    for (x_bomb, y_bomb), timer in bombs:
        if timer == 0:
            trigger_explosion((x_bomb, y_bomb), next_state)
        else:
            new_bombs.append(((x_bomb, y_bomb), timer - 1))

    if bomb_drop:
        new_bombs.append(((x, y), s.BOMB_TIMER))

    next_state['bombs'] = new_bombs
    
    for x in range(len(explosion_map)):
        for y in range(len(explosion_map[0])):
            if explosion_map[x][y] > 0:
                explosion_map[x][y] -= 1

    return next_state


def trigger_explosion(position, game_state):
    x, y = position
    explosion_range = s.BOMB_POWER
    directions = DIRECTIONS
    
    for dx, dy in directions:
        for step in range(1, explosion_range + 1):
            nx, ny = x + dx * step, y + dy * step
            if not (0 <= nx < len(game_state['field']) and 0 <= ny < len(game_state['field'][0])):
                break
            if game_state['field'][nx][ny] == -1:
                break
            game_state['explosion_map'][nx][ny] = s.EXPLOSION_TIMER
            if game_state['field'][nx][ny] == 1:
                game_state['field'][nx][ny] = 0
            
            game_state['others'] = [other for other in game_state['others'] if other[-1] != (nx, ny)]

            
def is_safe_to_drop_bomb(game_state):
    next_state = predict_next_state(game_state, True)
    field, _, bombs, _, _, _ = (
        next_state['field'], next_state['explosion_map'], next_state['bombs'],
        next_state['coins'], next_state['others'], next_state['self'][-1]
    )
    danger = danger_map(field, bombs)
    safe_positions = find_safe_positions(next_state, danger)
    if not safe_positions:
        return False
    return True


def find_crates_neighbors(field):
    center = field[1:-1, 1:-1]
    up = field[:-2, 1:-1]
    down = field[2:, 1:-1]
    left = field[1:-1, :-2]
    right = field[1:-1, 2:]
    result = (center == 0) & ((up == 1) | (down == 1) | (left == 1) | (right == 1))
    positions = np.argwhere(result)
    return positions + 1  # back to ori index


def look_for_target(game_state, features, safe_positions):
    next_state = predict_next_state(game_state)
    field, _, bombs, coins, others, (x, y) = (
        next_state['field'], next_state['explosion_map'], next_state['bombs'],
        next_state['coins'], next_state['others'], next_state['self'][-1]
    )
    danger = danger_map(field, bombs)
    candidates = np.array(list(safe_positions.keys()))
    directions = [(x+d[0], y+d[1]) for d in DIRECTIONS_INCLUDING_WAIT]
    weights = {"crates": 1, "coins": 50, "enemy": 1}

    if not candidates.size:
        return

    # Precompute positions for crates, coins, and enemies
    object_positions = {
        "crates": find_crates_neighbors(field),
        "coins": np.array(coins),
        "enemy": np.array([op[-1] for op in others])
    }

    scores = np.zeros(len(candidates))
    # Precompute distances from current positions to all types of objects
    for object_name, positions in object_positions.items():
        if positions.size:
            for i, index in enumerate(candidates):
                direction = np.array(directions[index])
                distances = calculate_steps(next_state, direction, positions, danger)
                if distances.size:
                    scores[i] += np.sum(weights[object_name] / (distances + 1))
                    if danger[x][y] > 0:
                        scores[i] += len(safe_positions[index]) * 20

    # Determine the best candidates with the maximum score
    best_indices = candidates[scores == scores.max()]

    if best_indices.size:
        chosen_index = np.random.choice(best_indices)
        features[chosen_index] = "target"


def find_safe_positions(game_state, danger, max_steps=5):
    field, explosion_map, bombs, _, others, pos = (
        game_state['field'], game_state['explosion_map'], game_state['bombs'],
        game_state['coins'], game_state['others'], game_state['self'][-1]
    )
    directions = DIRECTIONS_INCLUDING_WAIT
    first_step_to_safes = {}
    q = deque()
    q.append((pos[0], pos[1], 0, None))
    while q:
        x, y, steps, first_step = q.popleft()
        if (danger[x][y] > 0 and s.BOMB_TIMER - danger[x][y] < steps <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER) \
            or explosion_map[x][y] > steps: # explosion while agent passing by
            continue
        if steps == max_steps:
            if first_step not in first_step_to_safes:
                first_step_to_safes[first_step] = []
            if (x, y) not in first_step_to_safes[first_step]:
                first_step_to_safes[first_step].append((x, y))
            continue
        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] != 0:
                    continue
                if any((nx, ny) == other[-1] for other in others) and steps == 0:
                    continue
                bomb_prevent = any((nx, ny) == (bx, by) and timer >= steps for (bx, by), timer in bombs)
                if bomb_prevent:
                    continue
                new_first_step = i if first_step is None else first_step
                if steps < max_steps:
                    q.append((nx, ny, steps + 1, new_first_step))
    return first_step_to_safes


def danger_map(field: np.array, bombs: list) -> np.array:
    danger = np.zeros(field.shape)
    
    for (x_bomb, y_bomb), timer in bombs:
        update_danger(danger, field, x_bomb, y_bomb, timer)
        
    return danger


def update_danger(danger, field, x_bomb, y_bomb, timer):
    max_timer_value = s.BOMB_TIMER
    directions = DIRECTIONS
    
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
    directions = [(x+d[0], y+d[1]) for d in DIRECTIONS_INCLUDING_WAIT] # UP, RIGHT, DOWN, LEFT, WAIT
    danger = danger_map(field, bombs)
    field_labels = {1: 'crate', -1: 'wall', 0: 'free'}
    
    for i, j in directions:
        tile = field_labels[field[i, j]]
        if tile == 'free':
            if explosion_map[i, j] > 0:
                tile = 'wall'
            elif any(op[-1] == (i, j) for op in others):
                tile = 'enemy'
            elif any((b[0] == i and b[1] == j) for b, _ in bombs):
                tile = 'bomb'
            elif danger[i, j] > 0:
                tile = 'danger'
            elif any((coin[0] == i and coin[1] == j) for coin in coins):
                tile = 'coin'
        features.append(tile)
    if 'coin' not in features[:4]:
        safe_positions = find_safe_positions(game_state, danger)
        look_for_target(game_state, features, safe_positions)
    features.append(str(game_state['self'][2] & is_safe_to_drop_bomb(game_state)))  # Append bomb_left

    return features

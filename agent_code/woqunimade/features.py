import numpy as np
import copy
from collections import deque
import settings as s

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # UP, RIGHT, DOWN, LEFT
DIRECTIONS_INCLUDING_WAIT = DIRECTIONS + [(0, 0)]


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
        if (danger[x][y] > 0 and s.BOMB_TIMER - danger[x][y] < steps <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER) \
            or explosion_map[x][y] >= steps > 0: # explosion while agent passing by
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
    
    _, explosion_map, bombs, _, _, (x, y) = (
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
        new_bombs.append(((x, y), s.BOMB_TIMER-1))

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
    
    game_state['explosion_map'][x][y] = s.EXPLOSION_TIMER
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

            
def is_safe_to_drop_bomb(game_state):
    next_state = predict_next_state(game_state, True)
    field, _, bombs, _, _, (x, y) = (
        next_state['field'], next_state['explosion_map'], next_state['bombs'],
        next_state['coins'], next_state['others'], next_state['self'][-1]
    )
    danger = danger_map(field, bombs)
    crate = False
    safe = True

    safe_positions = find_safe_positions(next_state, danger)
    if not safe_positions:
        safe = False
        return safe, crate

    explosion_range = s.BOMB_POWER
    directions = DIRECTIONS

    for (dx, dy) in directions:
        for step in range(1, explosion_range + 1):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < danger.shape[0] and 0 <= ny < danger.shape[1]:
                if field[nx, ny] == -1:
                    break
                elif field[nx, ny] == 1:
                    crate = True
                    break
        if crate:
            break

    return safe, crate

def calculate_bomb_impact_matrix(field):
    height, width = field.shape
    bomb_impact_matrix = np.zeros((height * width, height * width), dtype=int)

    for x in range(height):
        for y in range(width):
            if field[x, y] == -1:
                continue
            index = x * width + y
            # Explore the bomb range considering BOMB_POWER
            for d in DIRECTIONS:
                for steps in range(1, s.BOMB_POWER + 1):
                    nx, ny = x + d[0] * steps, y + d[1] * steps
                    if 0 <= nx < height and 0 <= ny < width:
                        if field[nx, ny] == -1:
                            break
                        bomb_impact_matrix[index, nx * width + ny] = 1
    return bomb_impact_matrix


def find_crates_neighbors(field):
    bomb_impact_matrix = calculate_bomb_impact_matrix(field)
    height, width = field.shape
    crate_vector = (field == 1).astype(int).flatten()
    crate_counts = bomb_impact_matrix.dot(crate_vector)
    crate_counts_matrix = crate_counts.reshape(height, width)
    result_matrix = np.where(field == 0, crate_counts_matrix, 0)

    positions = np.argwhere(result_matrix)
    crates_amount = np.array([int(result_matrix[tuple(idx)]) for idx in positions])
    return positions, crates_amount


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
    # Precompute distances from current positions to coins and enemy
    for object_name in ['coins', 'enemy']:
        positions = object_positions[object_name]
        if positions.size:
            for i, index in enumerate(candidates):
                direction = np.array(directions[index])
                distances = calculate_steps(next_state, direction, positions, danger)
                if distances.size:
                    scores[i] += np.sum(weights[object_name] / (distances + 1))

    # crates
    object_name = 'crates'
    positions, amount = object_positions[object_name]
    if positions.size:
        for i, index in enumerate(candidates):
            direction = np.array(directions[index])
            distances = calculate_steps(next_state, direction, positions, danger)
            if distances.size:
                scores[i] += np.sum(weights[object_name] * amount / (distances + 1))
    
    # escape_positions
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
            or explosion_map[x][y] >= steps > 0: # explosion while agent passing by
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
    
    if game_state['self'][2]:
        safe, crate = is_safe_to_drop_bomb(game_state)
    else:
        safe, crate = False, False
    features.append(str(safe))
    features.append(str(crate))
    features.append(str('enemy' in features)) # block and make others in danger is a valuable attack
    
    return features

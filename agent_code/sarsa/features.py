import numpy as np
from collections import deque
import settings as s


def reachability(field, pos):
    """
    BFS
    """
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    reachable = np.zeros_like(field, dtype=bool)
    q = deque([pos])
    visited = set([tuple(pos)])

    while q:
        x, y = q.popleft()
        reachable[x][y] = True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
    
    return reachable


def is_dangerous(x, y, time_step, danger):
    start_time = danger[x][y]
    if start_time > 0 and start_time <= time_step < start_time + s.EXPLOSION_TIMER - 1:
        return True
    return False


def calculate_steps(field, danger, pos, objects, reachable):
    """
    BFS2
    You should:
        reachable = reachability(field, pos)
        dist = calculate_steps(field, danger, pos, objects, reachable)
    """
    distances = np.full((len(objects),), np.inf)
    target_index_map = {tuple(obj): idx for idx, obj in enumerate(objects) if reachable[obj[0]][obj[1]]}
    if not len(target_index_map):
        return distances
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    q = deque()
    q.append((pos[0], pos[1], 0))
    visited = set()
    visited.add((pos[0], pos[1], 0))

    while q:
        x, y, time_step = q.popleft()
        if (x, y) in target_index_map:
            idx = target_index_map[(x, y)]
            distances[idx] = min(distances[idx], time_step)
            del target_index_map[(x, y)]
            if not target_index_map:
                break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            next_time_step = time_step + 1
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx][ny] == 0 and not is_dangerous(nx, ny, next_time_step, danger) and (nx, ny, next_time_step) not in visited:
                    visited.add((nx, ny, next_time_step))
                    q.append((nx, ny, next_time_step))

    return distances


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
    features = ['0'] * 6

    return features
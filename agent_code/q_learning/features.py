import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import copy
import settings as s


DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # UP, RIGHT, DOWN, LEFT
DIRECTIONS_INCLUDING_WAIT = DIRECTIONS + [(0, 0)]


class Feature:
    # const
    bomb_impact_matrix: np.ndarray
    # var
    game_state: Dict
    feature: List[str]

    def __init__(self) -> None:
        # TODO: Init self.bomb_impact_matrix
        walls = np.zeros((s.COLS, s.ROWS), int)
        walls[:1, :] = -1
        walls[-1:, :] = -1
        walls[:, :1] = -1
        walls[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    walls[x, y] = -1

        self.bomb_impact_matrix = np.zeros((s.COLS * s.ROWS, s.COLS * s.ROWS), dtype=int)
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if walls[x, y] == -1:
                    continue
                index = x * s.ROWS + y
                self.bomb_impact_matrix[index, index] = 1
                # Explore the bomb range considering BOMB_POWER
                for d in DIRECTIONS:
                    for steps in range(1, s.BOMB_POWER + 1):
                        nx, ny = x + d[0] * steps, y + d[1] * steps
                        if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                            if walls[nx, ny] == -1:
                                break
                            self.bomb_impact_matrix[index, nx * s.ROWS + ny] = 1

    def calculate_danger_map(self, bombs) -> np.ndarray:
        bomb_map = np.zeros((s.COLS, s.ROWS), int)
        for (x_bomb, y_bomb), timer in bombs:
            bomb_map[x_bomb, y_bomb] = s.BOMB_TIMER - timer
        danger_map = self.bomb_impact_matrix.dot(bomb_map.flatten()).reshape((s.COLS, s.ROWS))
        return danger_map

    def find_safe_position(self, state, max_steps=5) -> List[int]:
        field, explosion_map, bombs, _, others, pos = (
            state['field'], state['explosion_map'], state['bombs'],
            state['coins'], state['others'], state['self'][-1]
        )
        danger = self.calculate_danger_map(bombs)
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
                    if not (field[nx, ny] == 0 or (field[nx, ny] == 1 and steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER)):
                        continue
                    if any((nx, ny) == other[-1] for other in others) and steps == 0:
                        continue
                    bomb_prevent = any((nx, ny) == (bx, by) and timer >= steps for (bx, by), timer in bombs)
                    if bomb_prevent:
                        continue
                    new_first_step = i if first_step is None else first_step
                    if steps < max_steps:
                        q.append((nx, ny, steps + 1, new_first_step))
        safe_position = [0] * 5
        for i in range(5):
            if i in first_step_to_safes:
                safe_position[i] = len(first_step_to_safes[i])
        return safe_position

    def predict_next_state(self, bomb_drop=False):
        next_state = copy.deepcopy(self.game_state)
        
        _, explosion_map, bombs, _, _, (x, y) = (
            next_state['field'], next_state['explosion_map'], next_state['bombs'],
            next_state['coins'], next_state['others'], next_state['self'][-1]
        )
        new_bombs = []

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

    def is_safe_to_drop_bomb(self) -> bool:
        next_state = self.predict_next_state(True)
        safe_positions = self.find_safe_position(next_state)
        if sum(safe_positions) == 0:
            return False
        return True

    def BFS(self, x, y) -> np.ndarray:
        distance = np.full_like(self.game_state['field'], np.inf)
        return distance

    def look_for_target(self) -> int:
        # use bfs
        pass

    def is_chance_to_attack(self) -> bool:
        pass

    def __call__(self, game_state: Dict) -> List[str]:
        self.game_state = game_state
        # TODO: Calculate feature list [up, right, down, left, center, bomb]
        # Step 1: Init what is in feature

        # Step 2: Find safe actions: List(Bool)

        # Step 3: Figure 'target' if feature[i] is free and safe
            # Step 3.1: Check if is a chance to attack
        return self.feature

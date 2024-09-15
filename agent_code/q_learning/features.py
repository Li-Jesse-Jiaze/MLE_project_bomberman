import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import copy
import settings as s


DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # UP, RIGHT, DOWN, LEFT
DIRECTIONS_INCLUDING_WAIT = DIRECTIONS + [(0, 0)]


class Feature:
    # const
    walls: np.ndarray
    bomb_impact_matrix: np.ndarray
    # var
    game_state: Dict
    features: List[str] = None

    main_enemy: str = None

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
        self.walls = walls

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
        danger_map = (self.bomb_impact_matrix * bomb_map.flatten()).max(1).reshape((s.COLS, s.ROWS))
        return danger_map

    def find_safe_position(self, state, max_steps=5) -> List[int]:
        field, explosion_map, bombs, others, pos = (
            state['field'], state['explosion_map'], state['bombs'], state['others'], state['self'][-1]
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
                    if not (field[nx, ny] == 0 or (field[nx, ny] == 1 and steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER and danger[nx][ny] > 0)):
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

    def predict_next_state(self, bomb_drop=None):
        next_state = copy.deepcopy(self.game_state)
        
        _, explosion_map, bombs, _, _, _ = (
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
            new_bombs.append((bomb_drop, s.BOMB_TIMER-1))
            next_state['self'] = (*next_state['self'][:-1], bomb_drop)

        next_state['bombs'] = new_bombs
        explosion_map -= (explosion_map > 0).astype(int)
        return next_state

    def is_safe_to_drop_bomb(self, pos) -> bool:
        next_state = self.predict_next_state(pos)
        safe_positions = self.find_safe_position(next_state)
        return sum(1 for s in safe_positions if s != 0)
        # if sum(safe_positions) == 0:
        #     return False
        # return True

    def BFS(self, game_state, x, y) -> np.ndarray:
        # distance = np.full(game_state['field'].shape, np.inf)
        # Consider unreachable coins and others
        m, n = game_state['field'].shape
        distance = np.abs(np.arange(m)[:, None] - x) + np.abs(np.arange(n) - y) + 2 * (s.BOMB_POWER + s.EXPLOSION_TIMER)
        field, explosion_map, bombs, _, others = (
            game_state['field'], game_state['explosion_map'], game_state['bombs'],
            game_state['coins'], game_state['others']
        )
        danger = self.calculate_danger_map(bombs)
        directions = DIRECTIONS_INCLUDING_WAIT

        q = deque()
        q.append((x, y, 0))

        visited = set()
        visited.add(tuple((x, y)))

        if bombs:
            max_wait_steps = max(bomb[-1] for bomb in bombs) + s.EXPLOSION_TIMER + 1
        else:
            max_wait_steps = explosion_map.max() + 1
        
        # BFS
        while q:
            x, y, steps = q.popleft()
            if (danger[x][y] > 0 and s.BOMB_TIMER - danger[x][y] < steps <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER) \
                or explosion_map[x][y] >= steps > 0: # explosion while agent passing by
                continue
            distance[x][y] = min(distance[x][y], steps)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and \
                    not (steps >= max_wait_steps and (nx, ny) in visited):
                    if not (field[nx, ny] == 0 or (field[nx, ny] == 1 and steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER and danger[nx][ny] > 0)):
                        continue
                    if any((nx, ny) == other[-1] for other in others) and steps == 0:
                        continue
                    bomb_prevent = any((nx, ny) == (bx, by) and timer >= steps for (bx, by), timer in bombs)
                    if bomb_prevent:
                        continue
                    visited.add((nx, ny))
                    q.append((nx, ny, steps + 1))
        return distance
    
    def find_crates_neighbors(self, field):
        height, width = field.shape
        crate_vector = (field == 1).astype(int).flatten()
        crate_counts = self.bomb_impact_matrix.dot(crate_vector)
        crate_counts_matrix = crate_counts.reshape(height, width)
        result_matrix = np.where(field == 0, crate_counts_matrix, 0)

        return result_matrix

    def look_for_target(self, safe: List[int], has_bomb: bool) -> int:
        if sum(safe) == 0:
            return -1
        next_state = self.predict_next_state()
        field, _, bombs, coins, others, (x, y) = (
            next_state['field'], next_state['explosion_map'], next_state['bombs'],
            next_state['coins'], next_state['others'], next_state['self'][-1]
        )
        danger = self.calculate_danger_map(bombs)
        directions = [(x+d[0], y+d[1]) for d in DIRECTIONS_INCLUDING_WAIT]
        weights_with_bomb = {"crates": 1, "coins": 200, "enemy0": 5, "enemies": -1, "escape": 35}
        weights_without_bomb = {"crates": 1, "coins": 200, "enemy0": -50, "enemies": -50, "escape": 35}
        weights = weights_with_bomb if has_bomb else weights_without_bomb

        coin_map = np.zeros_like(field)
        for coin in coins:
            coin_map[coin[0]][coin[1]] = 1
        enemy0_map = np.zeros_like(field)
        if self.main_enemy:
            enemy0 = {other[0]: other[-1] for other in others}[self.main_enemy]
            enemy0_map[enemy0[0]][enemy0[1]] = 1
        enemies_map = np.zeros_like(field)
        for other in others:
            enemies_map[other[-1][0]][other[-1][1]] = 1
        crate_map = self.find_crates_neighbors(field)

        score = {index: 0 for index in range(5) if safe[index]}
        for index in score.keys():
            nx, ny = directions[index]
            distance = self.BFS(next_state, nx, ny)
            coins_score = weights['coins'] * (coin_map / (distance + 1))
            enemy0_score = weights['enemy0'] * (enemy0_map / (distance + 1))
            enemies_score = weights['enemies'] * (enemies_map / (distance + 1))
            crates_score = weights['crates'] * (crate_map / (distance + 1))
            for i in range(crates_score.shape[0]):
                for j in range(crates_score.shape[1]):
                    if crates_score[i][j] > 0 and distance[i][j] < s.BOMB_TIMER:
                        crates_score[i][j] *= self.is_safe_to_drop_bomb((i, j))
            score[index] += np.sum(coins_score + enemy0_score + enemies_score + crates_score)

            # escape_positions
            if danger[x][y] > 0:
                score[index] += safe[index] * weights['escape']
        target = [index for index in score.keys() if score[index] == max(score.values())]
        target = np.random.choice(target)
        if target == 4:
            if crate_map[x][y] > 0 and self.is_safe_to_drop_bomb((x, y)) and has_bomb:
                target = 5
            else:
                score[4] = -np.inf
                if not any(score.values()):
                    return 4
                target = [index for index in score.keys() if score[index] == max(score.values())]
                target = np.random.choice(target)
        return target


    def is_chance_to_kill(self) -> bool:
        state_0 = copy.deepcopy(self.game_state)
        state_1 = self.predict_next_state(self.game_state['self'][-1])
        state_1['bombs'] = copy.deepcopy(self.game_state['bombs'])
        state_1['bombs'].append((self.game_state['self'][-1], s.BOMB_TIMER-1))
        for i in range(len(state_1['others'])):
            state_0['self'], state_0['others'][i] = state_0['others'][i], state_0['self']
            state_1['self'], state_1['others'][i] = state_1['others'][i], state_1['self']
            if sum(self.find_safe_position(state_0)) != 0 and sum(self.find_safe_position(state_1)) == 0 and self.find_safe_position(self.game_state)[-1]:
                return True
        return False

    def find_main_enemy(self):
        if len(self.game_state['others']):
            return min(
                self.game_state['others'],
                key=lambda other: abs(self.game_state['self'][-1][0] - other[-1][0]) + abs(self.game_state['self'][-1][1] - other[-1][1])
            )[0]
        return None

    def __call__(self, game_state: Dict) -> List[str]:
        self.game_state = game_state
        if len(game_state['others']):
            if self.main_enemy not in [o[0] for o in game_state['others']] or game_state['step'] % 30 == 0:
                self.main_enemy = self.find_main_enemy()
        else:
            self.main_enemy = None
        # Calculate feature list [up, right, down, left, center, bomb]
        # Step 1: Init what is in feature
        if game_state is None:
            return None
        
        field, explosion_map, bombs, coins, others, (x, y) = (
            game_state['field'], game_state['explosion_map'], game_state['bombs'],
            game_state['coins'], game_state['others'], game_state['self'][-1]
        )

        self.features = []
        directions = [(x+d[0], y+d[1]) for d in DIRECTIONS_INCLUDING_WAIT] # UP, RIGHT, DOWN, LEFT, WAIT
        field_labels = {1: 'block', -1: 'block', 0: 'free'}
        
        for i, j in directions:
            tile = field_labels[field[i, j]]
            if tile == 'free':
                if explosion_map[i, j] > 0:
                    tile = 'dead'
                elif any(op[-1] == (i, j) for op in others):
                    tile = 'enemy'
                elif any((b[0] == i and b[1] == j) for b, _ in bombs):
                    tile = 'block'
                elif any((coin[0] == i and coin[1] == j) for coin in coins):
                    tile = 'coin'
            self.features.append(tile)
        self.features.append(str(self.game_state['self'][2]))
        # Step 2: Find safe actions: List(Bool)
        safe = self.find_safe_position(self.game_state)
        for i in range(len(safe)):
            if safe[i] == 0 and self.features[i] in ['coin', 'free']:
                self.features[i] = 'dead'
        if not self.is_safe_to_drop_bomb((x, y)):
            self.features[-1] = 'dead'
        # safe.append(self.is_safe_to_drop_bomb(self.game_state['self'][-1]))
        # Step 3: Figure 'target' if feature[i] is free and safe
            # Step 3.1: Check if is a chance to attack
        target = self.look_for_target(safe, self.game_state['self'][2])
        if target >= 0:
            self.features[target] = 'target'
        if self.is_chance_to_kill() and self.features[-1] != 'dead':
            self.features[-1] = 'KILL!'
        return self.features

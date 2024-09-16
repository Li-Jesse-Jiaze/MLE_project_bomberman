import numpy as np
from collections import deque, defaultdict
from typing import List, Dict
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

    def __init__(self) -> None:
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
        categories = ['crate', 'wall', 'enemy', 'bomb', 'coin', 'free']
        self.category_to_index = {category: index for index, category in enumerate(categories)}

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
        first_step_to_safes = defaultdict(list)
        others_positions = set(other[-1] for other in others)
        bomb_dict = { (bx, by): timer for (bx, by), timer in bombs }
        q = deque()
        q.append((pos[0], pos[1], 0, None))
        while q:
            x, y, steps, first_step = q.popleft()
            if (danger[x][y] > 0 and s.BOMB_TIMER - danger[x][y] < steps <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER) \
                or explosion_map[x][y] >= steps > 0: # explosion while agent passing by
                continue
            if steps == max_steps:
                if (x, y) not in first_step_to_safes[first_step]:
                    first_step_to_safes[first_step].append((x, y))
                continue
            for i, (dx, dy) in enumerate(directions):
                nx, ny = x + dx, y + dy
                if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                    if not (field[nx, ny] == 0 or (field[nx, ny] == 1 and steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER and danger[nx][ny] > 0)):
                        continue
                    if (nx, ny) in others_positions and steps == 0:
                        continue
                    if (nx, ny) in bomb_dict and bomb_dict[(nx, ny)] >= steps:
                        continue
                    new_first_step = i if first_step is None else first_step
                    if steps < max_steps:
                        q.append((nx, ny, steps + 1, new_first_step))
        safe_position = [0] * 5
        for i in range(5):
            if i in first_step_to_safes:
                safe_position[i] = len(first_step_to_safes[i])
        return safe_position

    def trigger_explosion(self, position, game_state):
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

    def predict_next_state(self, bomb_drop=None):
        next_state = copy.deepcopy(self.game_state)
        
        _, explosion_map, bombs, _, _, _ = (
            next_state['field'], next_state['explosion_map'], next_state['bombs'],
            next_state['coins'], next_state['others'], next_state['self'][-1]
        )
        new_bombs = []

        for (x_bomb, y_bomb), timer in bombs:
            if timer == 0:
                self.trigger_explosion((x_bomb, y_bomb), next_state)
            else:
                new_bombs.append(((x_bomb, y_bomb), timer - 1))

        if bomb_drop:
            new_bombs.append((bomb_drop, s.BOMB_TIMER-1))
            next_state['self'] = (*next_state['self'][:-1], bomb_drop)

        next_state['bombs'] = new_bombs
        explosion_map -= (explosion_map > 0).astype(int)
        return next_state

    def is_safe_to_drop_bomb(self, pos):
        next_state = self.predict_next_state(pos)
        safe_positions = self.find_safe_position(next_state)
        return any(safe_positions)

    def BFS(self, game_state, x, y) -> np.ndarray:
        m, n = game_state['field'].shape
        distance = np.full((m, n), np.inf)
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
        
        other_positions = set(other[-1] for other in others)
        bomb_positions = {(bx, by): timer for (bx, by), timer in bombs}
        # BFS
        while q:
            x, y, steps = q.popleft()
            if (danger[x][y] > 0 and s.BOMB_TIMER - danger[x][y] < steps <= s.BOMB_TIMER - danger[x][y] + s.EXPLOSION_TIMER) or explosion_map[x][y] >= steps > 0: # explosion while agent passing by
                continue
            distance[x][y] = min(distance[x][y], steps)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and not (steps >= max_wait_steps and (nx, ny) in visited):
                    if not (field[nx, ny] == 0 or (field[nx, ny] == 1 and steps > s.BOMB_TIMER - danger[nx][ny] + s.EXPLOSION_TIMER and danger[nx][ny] > 0)):
                        continue
                    if (nx, ny) in other_positions and steps == 0:
                        continue
                    bomb_timer = bomb_positions.get((nx, ny))
                    if bomb_timer is not None and bomb_timer >= steps:
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

    def calculate_score_matrix(self, safe: List[int]) -> np.ndarray:
        score_matrix = np.zeros((5, 5))
        if not sum(safe):
            return score_matrix
        next_state = self.predict_next_state()
        field = next_state['field']
        coins = next_state['coins']
        others = next_state['others']
        x, y = next_state['self'][-1]  # current pos

        directions = [(x + dx, y + dy) for dx, dy in DIRECTIONS_INCLUDING_WAIT]
        crates_neighbors = self.find_crates_neighbors(field)
        coin_positions = np.array(coins)
        enemy_positions = [other[-1] for other in others]

        for index in range(5):
            if not safe[index]:
                continue
            nx, ny = directions[index]
            distances = self.BFS(next_state, nx, ny) + 1

            crates_score = crates_neighbors / distances

            # Adjust crates_score based on safety to drop a bomb
            mask = (crates_score > 0) & (distances < s.BOMB_TIMER)
            indices = np.argwhere(mask)
            for i, j in indices:
                crates_score[i, j] *= int(self.is_safe_to_drop_bomb((i, j)))
            score_matrix[index, 0] = np.sum(crates_score)

            if coin_positions.size > 0:
                coin_distances = distances[coin_positions[:, 0], coin_positions[:, 1]]
                score_matrix[index, 1] = (1 / coin_distances).sum()
            
            enemy_score = [0] * 3
            for enemy_idx, enemy_pos in enumerate(enemy_positions):
                ex, ey = enemy_pos
                enemy_distance = distances[ex, ey]
                enemy_score[enemy_idx] =  1 / enemy_distance
            score_matrix[index, 2:5] = enemy_score

        return score_matrix

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

    def __call__(self, game_state: Dict) -> List[str]:
        self.game_state = game_state
        # Calculate feature list [up, right, down, left, center, bomb]
        # Step 1: Init what is in feature
        if game_state is None:
            return None
        
        field, bombs, coins, others, (x, y) = (
            game_state['field'], game_state['bombs'],
            game_state['coins'], game_state['others'], game_state['self'][-1]
        )

        field_labels = {1: 'crate', -1: 'wall', 0: 'free'}
        directions = [(x + d[0], y + d[1]) for d in DIRECTIONS_INCLUDING_WAIT]
        features = np.zeros((5, 6))
        for index, (i, j) in enumerate(directions):
            tile = field_labels[field[i, j]]
            if any(op[-1] == (i, j) for op in others):
                tile = 'enemy'
            elif any((b[0] == i and b[1] == j) for b, _ in bombs):
                tile = 'bomb'
            elif any((coin[0] == i and coin[1] == j) for coin in coins):
                tile = 'coin'
            one_hot = np.zeros(6)
            one_hot[self.category_to_index[tile]] = 1
            # print(one_hot)
            features[index] = one_hot

        safe = self.find_safe_position(self.game_state)
        score_matrix = self.calculate_score_matrix(safe)
        score_and_safe = np.hstack((score_matrix, np.array([safe]).T))
        features = np.hstack((features[:, :-1], score_and_safe))
        bomb_valid = int(self.is_safe_to_drop_bomb((x, y)) and game_state['self'][2])
        kill = int(self.is_chance_to_kill())
        return features, bomb_valid, kill

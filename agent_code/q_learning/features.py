import numpy as np
from collections import deque
from typing import List, Dict, Tuple
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
                # Explore the bomb range considering BOMB_POWER
                for d in DIRECTIONS:
                    for steps in range(1, s.BOMB_POWER + 1):
                        nx, ny = x + d[0] * steps, y + d[1] * steps
                        if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                            if walls[nx, ny] == -1:
                                break
                            self.bomb_impact_matrix[index, nx * s.ROWS + ny] = 1

    def calculate_danger_map(self) -> np.ndarray:
        pass

    def find_safe_position(self) -> List[bool]:
        pass

    def is_safe_to_drop_bomb(self) -> bool:
        pass

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

import numpy as np

ROTATION_MAPPING = {
    0: {'UP': 'UP', 'RIGHT': 'RIGHT', 'DOWN': 'DOWN', 'LEFT': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    90: {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    180: {'UP': 'DOWN', 'RIGHT': 'LEFT', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
    270: {'UP': 'LEFT', 'RIGHT': 'UP', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'WAIT': 'WAIT', 'BOMB': 'BOMB'},
}

# Action flipping mapping (for horizontal and vertical flips)
VERTICAL_FLIP_MAPPING = {'UP': 'DOWN', 'RIGHT': 'RIGHT', 'DOWN': 'UP', 'LEFT': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
HORIZONTAL_FLIP_MAPPING = {'UP': 'UP', 'RIGHT': 'LEFT', 'DOWN': 'DOWN', 'LEFT': 'RIGHT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}


def adjust_action(action, rotation=0, flip_horizontal=False, flip_vertical=False):
    # Adjust action based on rotation
    action = ROTATION_MAPPING[rotation][action]
    if flip_horizontal:
        action = HORIZONTAL_FLIP_MAPPING[action]
    if flip_vertical:
        action = VERTICAL_FLIP_MAPPING[action]

    return action


def adjust_features(features, rotation=0, flip_horizontal=False, flip_vertical=False):
    if features is None or features.size == 0:
        return features
    
    matrix_flat, bombs = features[:-2], features[-2:]
    matrix = matrix_flat.reshape((5, -1))

    direction_index = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'CENTER': 4}
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    rotation = (rotation // 90) % 4
    directions = directions[-rotation:] + directions[:-rotation]

    if flip_horizontal:
        directions[1], directions[3] = directions[3], directions[1]

    if flip_vertical:
        directions[0], directions[2] = directions[2], directions[0]

    new_matrix = np.empty_like(matrix)
    for i, direct in enumerate(directions):
        new_matrix[i, :] = matrix[direction_index[direct], :]
    new_matrix[4, :] = matrix[direction_index['CENTER'], :]

    return np.concatenate((new_matrix.flatten(), bombs))
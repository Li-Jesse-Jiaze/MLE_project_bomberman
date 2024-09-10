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


def adjust_state(state, rotation=0, flip_horizontal=False, flip_vertical=False):
    if not len(state):
        return state

    direction_index = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # Define the initial order of directions before rotation

    # Normalize and apply clockwise rotation
    rotation = (rotation // 90) % 4  # Normalize the rotation to one of 0, 1, 2, or 3
    directions = directions[-rotation:] + directions[:-rotation]  # Rotate the list using slicing

    # Handle horizontal flip
    if flip_horizontal:
        directions[1], directions[3] = directions[3], directions[1]  # Swap 'RIGHT' and 'LEFT'

    # Handle vertical flip
    if flip_vertical:
        directions[0], directions[2] = directions[2], directions[0]  # Swap 'UP' and 'DOWN'

    # Update the state according to the new direction list
    new_state = [0] * 8
    for i, dir in enumerate(directions):
        new_state[i] = state[direction_index[dir]]
    new_state[4] = state[4]  # 'CENTER' remains unchanged
    new_state[5] = state[5]  # 'BOMBS_LEFT' remains unchanged
    new_state[6] = state[6]  # 'CRATE_IN_BOMB' remains unchanged
    new_state[7] = state[7]  # 'ENEMY_IN_BOMB' remains unchanged

    return new_state
import heapq

GOAL_STATE = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 0)
)

GOAL_POSITIONS = {
    tile: (r, c)
    for r, row in enumerate(GOAL_STATE)
    for c, tile in enumerate(row)
    if tile != 0
}

def manhattan_distance(state):
    distance = 0
    for r in range(4):
        for c in range(4):
            tile = state[r][c]
            if tile != 0:
                goal_r, goal_c = GOAL_POSITIONS[tile]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def find_blank(state):
    for r in range(4):
        for c in range(4):
            if state[r][c] == 0:
                return r, c

def get_neighbors(state):
    r, c = find_blank(state)
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            state_list = [list(row) for row in state]
            state_list[r][c], state_list[nr][nc] = state_list[nr][nc], state_list[r][c]
            new_state = tuple(tuple(row) for row in state_list)
            neighbors.append(new_state)

    return neighbors

def is_solvable(state):
    flat = [tile for row in state for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    blank_row, _ = find_blank(state)
    blank_row_from_bottom = 4 - blank_row
    if blank_row_from_bottom % 2 == 1:
        return inversions % 2 == 0
    else:
        return inversions % 2 == 1

def a_star_search(start_state):
    if not is_solvable(start_state):
        return None

    initial_h = manhattan_distance(start_state)
    open_set = [(initial_h, 0, start_state, [start_state])]
    g_scores = {start_state: 0}

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == GOAL_STATE:
            return path

        if g > g_scores.get(current, float('inf')):
            continue

        for neighbor in get_neighbors(current):
            tentative_g = g + 1
            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                h = manhattan_distance(neighbor)
                f_score = tentative_g + h
                heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))

    return None

def print_board(state):
    for row in state:
        print(" ".join(f"{val:2}" if val != 0 else " _" for val in row))
    print()

if __name__ == "__main__":
    start = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 0, 11),
        (13, 14, 15, 12)
    )

    print("Initial State:")
    print_board(start)

    solution_path = a_star_search(start)

    if solution_path is None:
        print("This puzzle configuration cannot be solved.")
    else:
        print(f"Optimal solution found in {len(solution_path) - 1} moves:\n")
        for step, state in enumerate(solution_path):
            print(f"Step {step}:")
            print_board(state)
from heapq import heappop, heappush
from math import gcd


State = tuple[int, int]


def solve_water_jug(
    capacity_a: int, capacity_b: int, target: int
) -> list[tuple[str, State]] | None:
    """Return the steps from the empty state to a state containing target.

    Each step is a pair of the action description and the resulting state.
    Return None if the target cannot be measured with the given jugs.
    """
    if capacity_a < 0 or capacity_b < 0 or target < 0:
        raise ValueError("Jug capacities and target must be non-negative.")

    start = (0, 0)
    if target == 0:
        return [("Start", start)]

    divisor = gcd(capacity_a, capacity_b)
    if target > max(capacity_a, capacity_b) or divisor == 0 or target % divisor:
        return None

    def heuristic(state: State) -> int:
        # One action changes either jug by at most the larger jug capacity.
        distance = min(abs(state[0] - target), abs(state[1] - target))
        largest_capacity = max(capacity_a, capacity_b)
        return (distance + largest_capacity - 1) // largest_capacity

    def successors(state: State) -> list[tuple[str, State]]:
        amount_a, amount_b = state
        transfer_a_to_b = min(amount_a, capacity_b - amount_b)
        transfer_b_to_a = min(amount_b, capacity_a - amount_a)
        return [
            ("Fill jug A", (capacity_a, amount_b)),
            ("Fill jug B", (amount_a, capacity_b)),
            ("Empty jug A", (0, amount_b)),
            ("Empty jug B", (amount_a, 0)),
            (
                "Pour jug A into jug B",
                (amount_a - transfer_a_to_b, amount_b + transfer_a_to_b),
            ),
            (
                "Pour jug B into jug A",
                (amount_a + transfer_b_to_a, amount_b - transfer_b_to_a),
            ),
        ]

    frontier: list[tuple[int, int, State]] = []
    heappush(frontier, (heuristic(start), 0, start))
    best_cost = {start: 0}
    previous: dict[State, tuple[State, str]] = {}

    while frontier:
        _, cost, state = heappop(frontier)
        if cost != best_cost.get(state):
            continue

        if target in state:
            path: list[tuple[str, State]] = []
            while state != start:
                parent, action = previous[state]
                path.append((action, state))
                state = parent
            path.append(("Start", start))
            path.reverse()
            return path

        for action, next_state in successors(state):
            next_cost = cost + 1
            if next_cost >= best_cost.get(next_state, float("inf")):
                continue

            best_cost[next_state] = next_cost
            previous[next_state] = (state, action)
            priority = next_cost + heuristic(next_state)
            heappush(frontier, (priority, next_cost, next_state))

    return None


if __name__ == "__main__":
    capacity_a = int(input("Capacity of jug A: "))
    capacity_b = int(input("Capacity of jug B: "))
    target = int(input("Target amount: "))

    solution = solve_water_jug(capacity_a, capacity_b, target)
    if solution is None:
        print("No solution exists for these capacities and target.")
    else:
        for step, (action, state) in enumerate(solution):
            print(f"{step}: {action} -> Jug A = {state[0]}, Jug B = {state[1]}")

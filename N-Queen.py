def solve_n_queens(n: int) -> list[int] | None:
    """Return one solution as a list of queen columns, one per row."""
    if n < 1:
        raise ValueError("N must be a positive integer.")

    queen_columns: list[int] = []

    def search(row: int) -> bool:
        if row == n:
            return True

        for column in range(n):
            if any(
                previous_column == column
                or abs(previous_column - column) == row - previous_row
                for previous_row, previous_column in enumerate(queen_columns)
            ):
                continue

            queen_columns.append(column)
            if search(row + 1):
                return True
            queen_columns.pop()

        return False

    return queen_columns.copy() if search(0) else None


def print_board(solution: list[int]) -> None:
    """Print a solution with Q marking each queen."""
    n = len(solution)
    for queen_column in solution:
        print(" ".join("Q" if column == queen_column else "." for column in range(n)))


if __name__ == "__main__":
    try:
        board_size = int(input("Enter N: "))
        solution = solve_n_queens(board_size)
    except ValueError as error:
        print(f"Invalid input: {error}")
    else:
        if solution is None:
            print(f"No solution exists for N = {board_size}.")
        else:
            print(f"One solution for N = {board_size}:")
            print_board(solution)

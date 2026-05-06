import sys
import time
from typing import Optional

# player symbols
PLAYER_B = 'B'
PLAYER_W = 'W'
EMPTY = '_'
ORIGIN = 'o'
INF = float('inf')


def default_board(rows=8, cols=8):
    board = [[EMPTY] * cols for _ in range(rows)]
    for c in range(cols):
        board[0][c] = PLAYER_B
        board[1][c] = PLAYER_B
        board[rows - 2][c] = PLAYER_W
        board[rows - 1][c] = PLAYER_W
    return board


def parse_board(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return [line.split() for line in lines]


def board_copy(board):
    return [row[:] for row in board]


def board_to_str(board):
    return '\n'.join(' '.join(row) for row in board)


def opponent(player):
    return PLAYER_W if player == PLAYER_B else PLAYER_B


def direction(player):
    # B goes down, W up
    return 1 if player == PLAYER_B else -1


def get_moves(board, player):
    rows = len(board)
    cols = len(board[0])
    dr = direction(player)
    opp = opponent(player)
    moves = []

    for r in range(rows):
        for c in range(cols):
            if board[r][c] != player:
                continue
            nr = r + dr
            if nr < 0 or nr >= rows:
                continue

            # straight forward
            if board[nr][c] in (EMPTY, ORIGIN):
                moves.append((r, c, nr, c))

            # diagonal left
            if c - 1 >= 0:
                t = board[nr][c - 1]
                if t in (EMPTY, ORIGIN) or t == opp:
                    moves.append((r, c, nr, c - 1))

            # diagonal right
            if c + 1 < cols:
                t = board[nr][c + 1]
                if t in (EMPTY, ORIGIN) or t == opp:
                    moves.append((r, c, nr, c + 1))

    return moves


def apply_move(board, move, player):
    fr, fc, tr, tc = move
    nb = board_copy(board)
    for r in range(len(nb)):
        for c in range(len(nb[0])):
            if nb[r][c] == ORIGIN:
                nb[r][c] = EMPTY
    nb[tr][tc] = player
    nb[fr][fc] = ORIGIN
    return nb


def is_terminal(board, rows, cols):
    for c in range(cols):
        if board[rows - 1][c] == PLAYER_B:
            return PLAYER_B
        if board[0][c] == PLAYER_W:
            return PLAYER_W

    b = sum(cell == PLAYER_B for row in board for cell in row)
    w = sum(cell == PLAYER_W for row in board for cell in row)
    if b == 0:
        return PLAYER_W
    if w == 0:
        return PLAYER_B
    return None


# --- heuristics ---

def heuristic_advance(board, player):
    rows = len(board)
    score = 0.0
    for r, row in enumerate(board):
        for cell in row:
            if cell == PLAYER_B:
                score += r              # B wants high row index
            elif cell == PLAYER_W:
                score -= (rows - 1 - r) # W wants low row index
    return score


def heuristic_safety(board, player):
    rows = len(board)
    cols = len(board[0])
    score = heuristic_advance(board, player)

    for r in range(rows):
        for c in range(cols):
            cell = board[r][c]
            if cell == PLAYER_B:
                # W captures B diagonally from above (r-1)
                cr = r - 1
                if cr >= 0:
                    if c - 1 >= 0 and board[cr][c - 1] == PLAYER_W:
                        score -= 2.0
                    if c + 1 < cols and board[cr][c + 1] == PLAYER_W:
                        score -= 2.0
            elif cell == PLAYER_W:
                # B captures W diagonally from below (r+1)
                cr = r + 1
                if cr < rows:
                    if c - 1 >= 0 and board[cr][c - 1] == PLAYER_B:
                        score += 2.0
                    if c + 1 < cols and board[cr][c + 1] == PLAYER_B:
                        score += 2.0
    return score


def heuristic_vanguard(board, player):
    rows = len(board)
    best_b = best_w = -1
    b_count = w_count = 0

    for r, row in enumerate(board):
        for cell in row:
            if cell == PLAYER_B:
                b_count += 1
                p = r                  # B wants high row
                if p > best_b:
                    best_b = p
            elif cell == PLAYER_W:
                w_count += 1
                p = rows - 1 - r       # W wants low row
                if p > best_w:
                    best_w = p

    return (best_b - best_w) * 5.0 + (b_count - w_count) * 1.0


HEURISTICS = {
    '1': heuristic_advance,
    '2': heuristic_safety,
    '3': heuristic_vanguard,
}


# --- search ---

class SearchStats:
    def __init__(self):
        self.nodes_visited = 0


def minimax_pure(board, depth, is_max, heuristic_fn, rows, cols, stats):
    stats.nodes_visited += 1

    winner = is_terminal(board, rows, cols)
    if winner == PLAYER_B:
        return INF
    if winner == PLAYER_W:
        return -INF
    if depth == 0:
        return heuristic_fn(board, PLAYER_B)

    player = PLAYER_B if is_max else PLAYER_W
    moves = get_moves(board, player)
    if not moves:
        return -INF if is_max else INF

    if is_max:
        value = -INF
        for move in moves:
            child = apply_move(board, move, player)
            value = max(value, minimax_pure(child, depth - 1, False, heuristic_fn, rows, cols, stats))
        return value
    else:
        value = INF
        for move in moves:
            child = apply_move(board, move, player)
            value = min(value, minimax_pure(child, depth - 1, True, heuristic_fn, rows, cols, stats))
        return value


def alphabeta(board, depth, alpha, beta, is_max, heuristic_fn, rows, cols, stats):
    stats.nodes_visited += 1

    winner = is_terminal(board, rows, cols)
    if winner == PLAYER_B:
        return INF
    if winner == PLAYER_W:
        return -INF
    if depth == 0:
        return heuristic_fn(board, PLAYER_B)

    player = PLAYER_B if is_max else PLAYER_W
    moves = get_moves(board, player)
    if not moves:
        return -INF if is_max else INF

    if is_max:
        value = -INF
        for move in moves:
            child = apply_move(board, move, player)
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False, heuristic_fn, rows, cols, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for move in moves:
            child = apply_move(board, move, player)
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True, heuristic_fn, rows, cols, stats))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def choose_move(board, player, depth, algorithm, heuristic_fn, rows, cols, stats):
    moves = get_moves(board, player)
    if not moves:
        return None

    is_max = (player == PLAYER_B)
    best_move = None
    best_value = -INF if is_max else INF

    for move in moves:
        child = apply_move(board, move, player)
        if algorithm == 'minimax':
            value = minimax_pure(child, depth - 1, not is_max, heuristic_fn, rows, cols, stats)
        else:
            value = alphabeta(child, depth - 1, -INF, INF, not is_max, heuristic_fn, rows, cols, stats)

        if is_max and value > best_value:
            best_value = value
            best_move = move
        elif not is_max and value < best_value:
            best_value = value
            best_move = move

    return best_move


def play_game(board, depth, algorithm, heuristic_fn):
    rows = len(board)
    cols = len(board[0])
    stats = SearchStats()
    current = PLAYER_B
    rounds = 0
    winner = None

    while True:
        winner = is_terminal(board, rows, cols)
        if winner:
            break

        move = choose_move(board, current, depth, algorithm, heuristic_fn, rows, cols, stats)
        if move is None:
            winner = opponent(current)
            break

        board = apply_move(board, move, current)
        rounds += 1
        current = opponent(current)

    return winner, rounds, board, stats


def main():
    if len(sys.argv) < 4:
        print("Usage: solution.py <algorithm> <heuristic> <depth>", file=sys.stderr)
        sys.exit(1)

    algorithm = sys.argv[1].lower()   # minimax or alphabeta
    heuristic_id = sys.argv[2]        # 1, 2, or 3
    depth = int(sys.argv[3])

    if heuristic_id not in HEURISTICS:
        print(f"Unknown heuristic: {heuristic_id}", file=sys.stderr)
        sys.exit(1)

    heuristic_fn = HEURISTICS[heuristic_id]

    raw = sys.stdin.read()
    board = parse_board(raw)

    t0 = time.perf_counter()
    winner, rounds, final_board, stats = play_game(board, depth, algorithm, heuristic_fn)
    elapsed = time.perf_counter() - t0

    # stdout: board then summary
    print(board_to_str(final_board))
    print(f"Rounds: {rounds} Winner: {winner}")

    # stderr: nodes then time
    print(stats.nodes_visited, file=sys.stderr)
    print(f"{elapsed:.3f}", file=sys.stderr)


if __name__ == '__main__':
    main()
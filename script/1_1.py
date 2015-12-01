import sys

WHITE_U = -0.04
WALL_U = 0
DISCOUNT_RATE = 0.99

def get_bellman(board, u_board, i, j):
    #Utility of the four directions
    actions = [u_board[i][j] for x in range(4)]
    
    if i > 0:
        if board[i - 1][j] != WALL_U:
            actions[0] = u_board[i - 1][j]
    if j > 0:
        if board[i][j - 1] != WALL_U:
            actions[1] = u_board[i][j - 1]
    if i < len(board[0]) - 1:
        if board[i + 1][j] != WALL_U:
            actions[2] = u_board[i + 1][j]
    if j < len(board[0]) - 1:
        if board[i][j + 1] != WALL_U:
            actions[3] = u_board[i][j + 1]

    max_u = max(actions)
    op_action = -1
    for i in range(4):
        if actions[i] == max_u:
            op_action = i
            break
    expected_u = 0.8 * max_u + 0.1 * actions[(op_action + 1) % 4] + 0.1 * actions[(op_action + 3) % 4]
    return board[i][j] + DISCOUNT_RATE * expected_u 

def get_initial_utility(board):
    u_board = [[0.0 for x in range(len(board[0]))] for x in range(len(board))]
    white_spaces = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != WHITE_U:
                u_board[i][j] = board[i][j]
            else:
                white_spaces += 1
    return u_board, white_spaces

def cal_MDP(board, terminal):
    u_board, white_spaces = get_initial_utility(board)
    converged = 0
    #Treat rewards as terminal states
    while converged < white_spaces:
        for i in range(len(board)):
            for j in range(len(board[0])):
                #print(u_board[i][j])
                if terminal:
                    if board[i][j] != WHITE_U and board[i][j] != WALL_U:
                        continue
                if board[i][j] == WHITE_U:
                    bellman_u = get_bellman(board, u_board, i, j)
                    #print("bellman " + str(bellman_u))
                    print(u_board[i][j] - bellman_u)
                    if round(u_board[i][j], 2) == round(bellman_u, 2):
                        converged += 1
                    else:
                        u_board[i][j] = bellman_u
        #print_matrix(u_board)
        #print(u_board)
    return u_board

def build_board():
    return [[WHITE_U, -1, WHITE_U, WHITE_U, WHITE_U, WHITE_U],
            [WHITE_U, WHITE_U, WHITE_U, WALL_U, -1, WHITE_U],
            [WHITE_U, WHITE_U, WHITE_U, WALL_U, WHITE_U, 3],
            [WHITE_U, WHITE_U, WHITE_U, WALL_U, WHITE_U, WHITE_U],
            [WHITE_U, WHITE_U, WHITE_U, WHITE_U, WHITE_U, WHITE_U],
            [1, -1, WHITE_U, WALL_U, -1, -1]]

def print_matrix(matrix):
    for row in matrix:
        row_str = ""
        for cell in row:
            row_str += " & " + str(cell)
        row_str += "\\\\"
        print(row_str)

if __name__ == "__main__":
    board = build_board()
    u_matrix = cal_MDP(board, True)
    print_matrix(u_matrix)
    #u_matrix = cal_MDP(board, False)
    #print_matrix(u_matrix)
    

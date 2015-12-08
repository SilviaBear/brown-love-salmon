import matplotlib.pyplot as plt
import sys
import random
import math

WHITE_U = -0.04
WALL_U = -sys.maxsize
DISCOUNT = 0.99

ITERATIONS = 10000

START = (3, 1)

THRESHOLD = ITERATIONS / 500

def build_std_u_board():
    return [[1.71118993773, -1, 1.86177820123 , 1.88573948542 , 1.90954932741 , 2.34785851515],
        [2.122199664 , 2.19207216008 , 2.26218673589 , 0 , -1 , 2.48279689234],
        [2.1907661028 , 2.2702246294 , 2.35001907221 , 0 , 2.7445608611 , 3],
        [2.24899423819 , 2.3433775794 , 2.44017038605 , 0 , 2.79778956546 , 2.90009008544],
        [2.18336718311 , 2.28293328619 , 2.53358476954 , 2.68535354163 , 2.71925952069 , 2.80363822444],
        [1 , -1 , 2.07280703382 , 0 , -1 , -1]]

def print_matrix(matrix):
    for row in matrix:
        row_str = ""
        for cell in row:
            row_str += " & " + str(cell)
        row_str += "\\\\"
        print(row_str)

def learning_rate_func(t):
    return 20.0 / (19.0 + t)

def build_board():
    return [[WHITE_U, -1, WHITE_U, WHITE_U, WHITE_U, WHITE_U],
            [WHITE_U, WHITE_U, WHITE_U, WALL_U, -1, WHITE_U],
            [WHITE_U, WHITE_U, WHITE_U, WALL_U, WHITE_U, 3],
            [WHITE_U, WHITE_U, WHITE_U, WALL_U, WHITE_U, WHITE_U],
            [WHITE_U, WHITE_U, WHITE_U, WHITE_U, WHITE_U, WHITE_U],
            [1, -1, WHITE_U, WALL_U, -1, -1]]

def init_q_board():
    return [[[0 for x in range(4)] for x in range(6)] for x in range(6)]

def build_u_board(q_board):
    u_board = [[0 for x in range(6)] for x in range(6)]
    for x in range(6):
        for y in range(6):
            u_board[x][y] = max(q_board[x][y])
    return u_board

def move(pos, direction):
    next_pos = None
    if direction == 0:
        next_pos = (pos[0] - 1, pos[1])
    elif direction == 1:
        next_pos = (pos[0], pos[1] - 1)
    elif direction == 2:
        next_pos = (pos[0] + 1, pos[1])
    elif direction == 3:
        next_pos = (pos[0], pos[1] + 1)

    if next_pos[0] < 0:
        next_pos = (0, next_pos[1])
    if next_pos[0] > 5:
        next_pos = (5, next_pos[1])
    if next_pos[1] < 0:
        next_pos = (next_pos[0], 0)
    if next_pos[1] > 5:
        next_pos = (next_pos[0], 5)
    return next_pos

def get_action(cur_pos, q_board, n_board):
    selected = -1
    max_reward = -sys.maxsize
    x = cur_pos[0]
    y = cur_pos[1]
    for i in range(4):
        next_pos = move(cur_pos, i)
        _x = next_pos[0]
        _y = next_pos[1]
        next_reward = max(q_board[next_pos[0]][next_pos[1]])
        if q_board[next_pos[0]][next_pos[1]] > max_reward:
            if n_board != None and n_board[x][y][i] > THRESHOLD:
                continue
            selected = i
            max_reward = next_reward

    if selected == -1:
        selected = random.randint(0, 3)
        next_pos = move(cur_pos, selected)
        max_reward = max(q_board[next_pos[0]][next_pos[1]])
    return selected, max_reward

def update_TD(cur_pos, a, new_pos, r_board, q_board, time, max_reward):
    x = cur_pos[0]
    y = cur_pos[1]
    _x = new_pos[0]
    _y = new_pos[1]
    #print("original u " + str(q_board[x][y][a]))
    #print("max reward " + str(max_reward))
    q_board[x][y][a] = q_board[x][y][a] + learning_rate_func(time) * (r_board[x][y] + DISCOUNT * max_reward - q_board[x][y][a])
    #print("updated u " + str(q_board[x][y][a]))

def calculate_RMSE(u_board, _u_board):
    _sum = 0.0
    for x in range(6):
        for y in range(6):
            _sum += math.pow((u_board[x][y] - _u_board[x][y]), 2)

    return math.pow(_sum / 36, 0.5)
    
def learn():
    r_board = build_board()
    q_board = init_q_board()
    n_board = init_q_board()
    time = 0
    cur_pos = START
    num_trial = 0
    log_counter = 0
    RMSE = []
    u_estimates = [[0 for x in range(15)] for x in range(36)]
    while log_counter < 15:
        x = cur_pos[0]
        y = cur_pos[1]

        if r_board[x][y] != WHITE_U and r_board[x][y] != WALL_U:
            for i in range(4):
                q_board[x][y][i] = r_board[x][y]
                n_board[x][y][i] += 1
            cur_pos = START
            num_trial += 1
            if num_trial == math.pow(2, log_counter):
            
                u_board = build_u_board(q_board)
                _u_board = build_std_u_board()
                RMSE.append(calculate_RMSE(u_board, _u_board))
                for i in range(6):
                    for j in range(6):
                        print(i * 6 + j)
                        print(log_counter)
                        u_estimates[i * 6 + j][log_counter] = u_board[i][j]

                print_matrix(u_board)
                log_counter += 1

            continue
        
        a, max_reward = get_action(cur_pos, q_board, n_board)
        n_board[x][y][a] += 1
        new_pos = move(cur_pos, a)
        #print(cur_pos)
        #print(new_pos)
        #print(max_reward)
        #print(r_board[new_pos[0]][new_pos[1]])
        if r_board[new_pos[0]][new_pos[1]] == WALL_U:
            #print("bump wall")
            new_pos = cur_pos
            max_reward = max(q_board[x][y])
        update_TD(cur_pos, a, new_pos, r_board, q_board, time, max_reward)
        cur_pos = new_pos
        
    return RMSE, u_estimates
        
if __name__ == "__main__":
    RMSE, u_estimates = learn()
    print(RMSE)
    print(u_estimates)
    for estimates in u_estimates:
        plt.plot(estimates)
    plt.show()
    plt.plot(RMSE)
    plt.show()

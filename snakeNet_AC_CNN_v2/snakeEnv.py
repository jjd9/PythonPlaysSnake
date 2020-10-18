"""

"""

import numpy as np
import cv2
import time
import IPython, sys
from numba import njit


# snake action space
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3

# internal board size
BOARD_ROWS = 20 #40
BOARD_COLS = 20 #40

# dimensions used for rendering board
RENDER_BLOCK_SIZE = 15
RENDER_BOARD_ROWS = int(BOARD_ROWS*RENDER_BLOCK_SIZE)
RENDER_BOARD_COLS = int(BOARD_COLS*RENDER_BLOCK_SIZE)
RENDER_BLOCK_SPACING = 2

# game penalty
FOOD_REWARD = 1.0
EAT_SELF_PENALTY = -1.0
OUT_OF_BOUNDS_PENALTY = -1.0

# color palette 
BLACK = (0,0,0)
RED   = (0, 0, 255)
GREEN = (0,255,0)
BLUE = (255,0,0)
WHITE = (255, 255, 255)

# object identifiers
EMPTY_ID = 0.0
FOOD_ID = 1.0
HEAD_ID = 3.0
SNAKE_ID = 2.0
OBS_ID = 2.0

# @njit
# def np_apply_along_axis(func1d, axis, arr):
#   assert arr.ndim == 2
#   assert axis in [0, 1]
#   if axis == 0:
#     result = np.empty(arr.shape[1])
#     for i in range(len(result)):
#       result[i] = func1d(arr[:, i])
#   else:
#     result = np.empty(arr.shape[0])
#     for i in range(len(result)):
#       result[i] = func1d(arr[i, :])
#   return result

# @njit()
# def randomPosition(snake_body, rows, cols):
#     """Generate random point until it does not collide with the snake's body"""
#     row = np.random.randint(0,rows)
#     col = np.random.randint(0,cols)
#     new_point = np.array([col, row])
#     while np.any(np_apply_along_axis(np.all, 1, snake_body==new_point)):       
#         row = np.random.randint(0,rows)
#         col = np.random.randint(0,cols)
#         new_point = np.array([col, row])
#     return col, row



class Snake:
    def __init__(self, row, col):
        self.body = np.array([[row, col]])
        self.lastTailPosition = np.array([row, col])
        self.action_map = {
            UP: np.array([0, -1]),
            DOWN: np.array([0, 1]),
            LEFT: np.array([-1, 0]),
            RIGHT: np.array([1, 0]),
        }
        self.reward = 0

    def move(self, direction):
        """Move snake in given direction"""
        action = np.array(self.action_map[direction])
        if self.body.shape[0] > 1:
            if np.all(self.body[0,:] + action == self.body[1,:]):
                self.reward += 2*EAT_SELF_PENALTY
            else:
                self.body[1:,:] = np.array(self.body[:-1,:])

        self.lastTailPosition = np.array(self.body[-1,:])
        self.body[0,:] += np.array(action)

    def eatFood(self):
        """Snake eats food and grows 1 segment"""
        self.body = np.vstack((self.body, self.lastTailPosition))


class GridWorld:
    # rendering will be col, row --> x, y
    def __init__(self, grid_state=False):
        # offsets for generating snake observation vector
        self.rows = BOARD_ROWS
        self.cols = BOARD_COLS
        self.grid_state = grid_state

        if not self.grid_state:
            self.state = np.zeros((6,)).astype("float")
        else:
            self.state = np.full((3, self.rows, self.cols), EMPTY_ID)

        self.board = np.full((self.cols, self.rows),EMPTY_ID).astype("float")

        self.radius = None

        self.reset()

    def reset(self, radius=None):
        self.radius = radius
        # generate snake`
        self.snake = Snake(*self.randomPosition())
        size = 3
        direction = np.random.randint(0,4)
        for _ in range(size):
            self.snake.move(direction)
            self.snake.eatFood()
        min_x_dim = self.snake.body[:,0].min()
        max_x_dim = self.snake.body[:,0].max()
        min_y_dim = self.snake.body[:,1].min()
        max_y_dim = self.snake.body[:,1].max()
        if min_x_dim < 0:
            self.snake.body[:,0] -= min_x_dim - 5
        if min_y_dim < 0:
            self.snake.body[:,1] -= min_y_dim - 5
        if max_x_dim >= self.cols:
            dx = max_x_dim - self.cols
            self.snake.body[:,0] -= dx + 5
        if max_y_dim >= self.rows:
            dy = max_y_dim - self.rows
            self.snake.body[:,1] -= dy + 5

        # generate food
        self.food = np.array(self.randomPosition(self.radius))

        self.update_state()
        return self.state

    def seed(self, randomSeed):
        np.random.seed(randomSeed)

    def step(self, action):
        """Game takes a step based on given snake action"""
        
        # move snake
        self.snake.reward = 0
        self.snake.move(action)
        # see if snake ate food
        self.check_food_collision()
        done = self.check_out_of_bounds()        
        if not done:
            done = self.check_snake_collision()
            self.update_state(snake_collision=done)            
        else:
            self.snake.reward += OUT_OF_BOUNDS_PENALTY

        return self.state, self.snake.reward, done

    def randomPosition(self, radius=None):
        """Generate random point until it does not collide with the snake's body"""
        if hasattr(self, "snake"):
            if radius is None:                           
                row = np.random.randint(0,self.rows)
                col = np.random.randint(0,self.cols)
                new_point = np.array([col, row])
                while np.any(np.all(self.snake.body==new_point, axis=1)):       
                    row = np.random.randint(0,self.rows)
                    col = np.random.randint(0,self.cols)
                    new_point = np.array([col, row])
            else:
                min_y = max(0, self.snake.body[0,1]-radius)
                max_y = min(self.rows-1, self.snake.body[0,1]+radius)
                min_x = max(0, self.snake.body[0,0]-radius)
                max_x = min(self.cols-1, self.snake.body[0,0]+radius)

                row = np.random.randint(min_y,max_y)
                col = np.random.randint(min_x,max_x)
                new_point = np.array([col, row])
                tries = 0
                while np.any(np.all(self.snake.body==new_point, axis=1)):       
                    row = np.random.randint(min_y,max_y)
                    col = np.random.randint(min_x,max_x)
                    new_point = np.array([col, row])
                    tries+=1
                    if tries > 50:
                        col, row = self.randomPosition()
                        break

        else:
            row = np.random.randint(0,self.rows)
            col = np.random.randint(0,self.cols)
        return col, row

    def check_out_of_bounds(self):
        """Check if snake out of bounds"""
        if np.any(self.snake.body[:,0]>=self.cols):
            return True
        if np.any(self.snake.body[:,1]>=self.rows):
            return True
        if np.any(self.snake.body[:,0] < 0):
            return True
        if np.any(self.snake.body[:,1] < 0):
            return True
        return False

    def check_snake_collision(self):
        """Check if snake head collides with snake body"""
        if self.snake.body.shape[0] > 1:        
            snake_head = self.snake.body[0,:]
            collision = np.any(np.all(self.snake.body[1:,:]==snake_head,axis=1))
            if collision:
                self.snake.reward += EAT_SELF_PENALTY
        else:
            collision = False
        return collision

    def check_food_collision(self):
        """Check if snake head collides with food"""
        if np.all(self.snake.body[0,:] == self.food):
            # snake eats food and gets rewarded
            self.snake.eatFood()
            self.snake.reward += FOOD_REWARD
            # generate new food
            self.food = np.array(self.randomPosition(self.radius))

    def render(self, show=True):
        """Render game board"""

        # initialize board
        board = np.zeros((RENDER_BOARD_ROWS,RENDER_BOARD_COLS,3), np.uint8)

        # draw food
        start_point = tuple(((self.food)*RENDER_BLOCK_SIZE).astype(int)) 
        end_point   = tuple(((self.food+1.0)*RENDER_BLOCK_SIZE).astype(int))

        board = cv2.rectangle(board, start_point, end_point, RED, -1)

        # snake
        segment = self.snake.body[0,:]
        start_point = tuple(((segment)*RENDER_BLOCK_SIZE).astype(int)) 
        end_point   = tuple(((segment+1.0)*RENDER_BLOCK_SIZE).astype(int))
        board = cv2.rectangle(board, start_point, end_point, BLUE, -1)

        for segment in self.snake.body[1:,:]:
            start_point = tuple(((segment)*RENDER_BLOCK_SIZE).astype(int)) 
            end_point   = tuple(((segment+1.0)*RENDER_BLOCK_SIZE).astype(int))
            board = cv2.rectangle(board, start_point, end_point, GREEN, -1)

        if show:
            cv2.imshow("Snake Game", board)
            cv2.waitKey(1)
            time.sleep(0.01)
        return board


    def update_state(self, snake_collision=False):
        self.state = np.zeros((self.rows+2, self.cols+2,3), np.uint8)
        self.state[:] = WHITE
        self.state[1:self.rows+1, 1:self.cols+1] = BLACK
        self.state[self.food[0]+1, self.food[1]+1,:] = RED
        self.state[self.snake.body[1:,0]+1, self.snake.body[1:,1]+1,:] = GREEN
        self.state[self.snake.body[0,0]+1, self.snake.body[0,1]+1,:] = BLUE
        self.state = cv2.flip(self.state,1)
        self.state = np.rot90(self.state,1).reshape(3,self.rows+2,self.cols+2)
        

if __name__ == "__main__":
    # rot = 1
    act_90 = [3, 2, 0, 1]
    # rot = 2
    act_180 = [1, 0, 3, 2]
    # rot = 3
    act_270 = [2, 3, 1, 0]
    # flip
    act_flip = [1, 0, 2, 3]
    
    rotate_action_list = [act_90, act_180, act_270]

    flip = True
    rotation = 1
    env = GridWorld(grid_state=True)
    env.reset(radius=1)
    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    try:
        while True:
            env.render()
            a = env.state
            if flip:
                a = np.flip(a.reshape(22,22,3), 1).reshape(3,22,22)
            a = np.rot90(a.reshape(22,22,3), rotation)
            cv2.imshow("main",a)
            # w, a, s, d
            # 101, 115, 100, 102
            val = cv2.waitKeyEx()
            if val == 119:
                print("up")
                command=(UP)
            elif val == 97:
                print("left")
                command=(LEFT)
            elif val ==115:
                print("down")
                command=(DOWN)
            elif val ==100:
                print("right")
                command=(RIGHT)
            else:
                continue

            if flip:
                command = act_flip[command]

            command = rotate_action_list[rotation-1][command]

            state, reward, done = env.step(command)
            print("Reward: ", reward)
            if done:
                print("game over")
                break
    except KeyboardInterrupt:
        cv2.destroyAllWindows()        
    finally:
        cv2.destroyAllWindows()
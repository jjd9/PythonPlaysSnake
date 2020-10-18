"""

"""

import numpy as np
import cv2
import time
import IPython, sys

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

# object identifiers
EMPTY_ID = 0.0
FOOD_ID = 1.0
SNAKE_ID = 2.0
OBS_ID = 2.0

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
                self.reward -= 2 #action = self.body[0,:] - self.body[1,:]

        self.lastTailPosition = np.array(self.body[-1,:])

        self.body[1:,:] = np.array(self.body[:-1,:])
        self.body[0,:] += np.array(action)

    def eatFood(self):
        """Snake eats food and grows 1 segment"""
        assert(not np.any(np.all(self.body==self.lastTailPosition,axis=1))), "body overlap"
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
            self.state = np.full((1, self.rows, self.cols), EMPTY_ID)


        self.reset()

    def reset(self):
        # generate snake`
        self.snake = Snake(*self.randomPosition())

        size = 3 #np.random.randint(2,10)
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
        self.food = np.array(self.randomPosition())

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
            self.update_state()            
        else:
            self.snake.reward += OUT_OF_BOUNDS_PENALTY
        return self.state, self.snake.reward, done

    def randomPosition(self):
        """Generate random point until it does not collide with the snake's body"""
        if hasattr(self, "snake"):                           
            no_collision = False
            while not no_collision:       
                row = np.random.randint(0,self.rows)
                col = np.random.randint(0,self.cols)
                new_point = np.array([col, row])
                no_collision = not np.any(np.all(self.snake.body==new_point,axis=1))
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
            self.food = np.array(self.randomPosition())

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
        board = cv2.rectangle(board, start_point, end_point, (0,255,0), -1)

        for segment in self.snake.body[1:,:]:
            start_point = tuple(((segment)*RENDER_BLOCK_SIZE).astype(int)) 
            end_point   = tuple(((segment+1.0)*RENDER_BLOCK_SIZE).astype(int))
            board = cv2.rectangle(board, start_point, end_point, GREEN, -1)

        if show:
            cv2.imshow("Snake Game", board)
            cv2.waitKey(1)
            time.sleep(0.01)
        return board


    def update_state(self):
        board = np.full((self.cols, self.rows),EMPTY_ID).astype("float")
        board[self.food[0], self.food[1]] = FOOD_ID
        board[self.snake.body[:,0], self.snake.body[:,1]] = SNAKE_ID

        if self.grid_state:
            self.state[0,:,:] = board.copy()
        else:
            snake_head = self.snake.body[0,:]
            self.state[:2] = np.sign(self.food - snake_head).astype("float")
            self.state[2:] = 0.0
            if self.snake.body.shape[0] > 1:
                if snake_head[0]+1 <=  self.cols-1:             
                    self.state[2] =  board[snake_head[0]+1,snake_head[1]] #right

                if 0<snake_head[0]:
                    self.state[3] =  board[snake_head[0]-1,snake_head[1]] # left

                if snake_head[1]+1 <= self.rows-1:
                    self.state[4] =  board[snake_head[0],snake_head[1]+1] #  down

                if 0 < snake_head[1]:
                    self.state[5] =  board[snake_head[0],snake_head[1]-1] # up

            if snake_head[0]+1 >  self.cols-1:             
                self.state[2] = OBS_ID

            if 0==snake_head[0]:
                self.state[3] = OBS_ID

            if snake_head[1]+1 > self.rows-1:
                self.state[4] = OBS_ID

            if 0 == snake_head[1]:
                self.state[5] = OBS_ID


if __name__ == "__main__":
    env = GridWorld()
    env.reset()
    try:
        while True:
            env.render()
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
            state, reward, done = env.step(command)
            print(state)
            if done:
                print("game over")
                break
    finally:
        cv2.destroyAllWindows()
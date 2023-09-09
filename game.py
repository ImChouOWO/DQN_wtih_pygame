import pygame
import sys
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SCREEN_SIZE = [320, 400]
BAR_SIZE = [40, 5]
BALL_SIZE = [15, 15]
count = 0
class GameEnv(object):
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Simple Game DQN')
        self.reset()

    def reset(self):
        self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2
        self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2
        # self.ball_dir_x = random.choice([-1, 1])  
        self.ball_dir_x = 1  
        self.ball_dir_y = -1  
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
        self.score = 0
        self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
        return [self.ball_pos_x, self.ball_pos_y, self.ball_dir_x, self.ball_dir_y, self.bar_pos_x]



    #外部控制腳本 
    def step(self, action):
        global count
        if action == 0:
            self.bar_move_left()
        elif action == 1:
            self.bar_move_right()

        self.screen.fill(BLACK)
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(self.screen, WHITE, self.bar_pos)

        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 3
        pygame.draw.rect(self.screen, WHITE, self.ball_pos)

        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= SCREEN_SIZE[0]:
            self.ball_dir_x = self.ball_dir_x * -1

        reward = 0
        done = False
        if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
            # count +=1
            # reward = 1 + count
           
            
            self.score += 1
            reward = self.score
        elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
            if self.bar_pos.right <0 or self.bar_pos.left >SCREEN_SIZE[0]:
                reward =-2
            else:
                reward = -1
            done = True
            count = 0
        pygame.display.update()
        self.clock.tick(30)

        next_state = [self.ball_pos_x, self.ball_pos_y, self.ball_dir_x, self.ball_dir_y, self.bar_pos_x]
        return next_state, reward, done, {}

    def bar_move_left(self):
        self.bar_pos_x = self.bar_pos_x - 2

    def bar_move_right(self):
        self.bar_pos_x = self.bar_pos_x + 2

    def render(self):
        pass

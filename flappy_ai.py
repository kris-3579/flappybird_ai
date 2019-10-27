from __future__ import division, print_function
import collections
import pygame
import numpy as np
import random
import os


class flappy_game(object):

    def __init__(self):

        pygame.init()
        pygame.key.set_repeat(10, 100)

        # game const
        self.white = (255,255,255)
        self.black = (0,0,0)
        self.width = 400
        self.height = 600
        self.font_size = 30
        self.time = 0


        # bird const
        self.radius = 20
        self.num = 0
        self.b_y = 40
        self.b_x = 20
        self.vel = 4
        self.grav = 12
        self.lift = 10
        self.isjumping = False
        self.jump_count = 0
        self.hitbox = pygame.Rect(self.b_x, self.b_y, self.radius *2, self.radius * 2)

        # pipe const
        self.gap = 150
        self.w_width = 50
        self.pip_arr = []

        # ai const
        self.max_tries = 1
        self.custom_event = pygame.USEREVENT + 1
        self.font = pygame.font.SysFont("Comic Sans MS", self.font_size)

    def reset(self):

        self.frames = collections.deque(maxlen= 4)
        self.game_over = False
        self.reward = 0
        self.num_tries = 0
        self.game_score = 0

        self.win = pygame.display.set_mode((self.width,self.height))
        self.b_x = 20
        self.b_y = 40
        self.pip_arr.clear()
        self.time = 0
        self.clock = pygame.time.Clock()

    def step(self, action):
        pygame.event.pump()

        if action == 0:
            self.jump_count = 16
            self.isjumping = True

        if action == 1:
            pass

        self.win.fill(self.black)

        # bird fall
        self.b_y += int(self.grav) + self.vel

        # bird jump
        if self.isjumping:
            if self.jump_count >= -16:
                neg = 1
                if self.jump_count < 0:
                    neg = -1
                self.b_y -= int(self.jump_count ** 2 * 0.1 * neg)
                self.jump_count -= 1
            else:
                self.isjumping = False
                self.jump_count = 10


        # draw bird
        self.hitbox = pygame.Rect(self.b_x - self.radius, self.b_y - self.radius, self.radius * 2, self.radius * 2)
        pygame.draw.circle(self.win, self.white, (self.b_x, self.b_y), self.radius)

        # die bird
        if self.b_y + self.radius >= self.height:
            self.num_tries += 1
            self.reward = -10

        if self.b_y + self.radius <= 0:
            self.num_tries += 1
            self.reward = -10


        # add pipe
        if ((self.time % 150) < 1):
            x = random.randrange(self.height - self.gap)
            self.pip_arr.append([self.width, 0, self.w_width, x])
            self.pip_arr.append([self.width, x + self.gap, self.w_width, self.height - x - self.gap])

        # pipes
        for pipe in self.pip_arr:

            # delete
            if pipe[0] + self.width < 0:
                self.pip_arr.remove(pipe)

            # move
            pipe[0] -= 2

            # draw
            pygame.draw.rect(self.win, self.white, pipe)

            # score
            if self.b_x + self.radius == self.w_width + pipe[0]:
                self.game_score += 1
                self.reward = 10

            # crash
            tmp_rect = pygame.Rect(pipe)
            if tmp_rect.colliderect(self.hitbox):
                self.num_tries += 1
                self.reward = -10

        score_text = self.font.render("Score: {:d}/{:d}, Ball: {:d}"
                                          .format(self.game_score,
                                                  self.max_tries, self.num),
                                          True, self.white)
        self.time += 1
        pygame.display.flip()
        pygame.display.update()
        if self.num_tries >= self.max_tries:
            self.game_over = True

        self.frames.append(pygame.surfarray.array2d(self.win))

        self.clock.tick(30)
        return self.get_frames(), self.reward, self.game_over

    def get_frames(self):
        return np.array(list(self.frames))




















import pygame
import sys
import time
import random
from pygame.locals import QUIT
import numpy as np
import gym
from gym import spaces
import threading
from tensorflow import keras

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

class SquirrelEnv(gym.Env):
    metadata = {"render.modes": ['human']}

    def __init__(self):
        self.frame_size_x = 800
        self.frame_size_y = 700
        self.action_space = spaces.Discrete(4)
        pygame.init()
        self.observation_space = spaces.Box(low=0, high=255, shape=(800, 700, 3), dtype=np.uint8)
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        self.reset()
        self.STEP_LIMIT = 1000
        self.sleep = 0

    @staticmethod
    def change_direction(action, direction):
        if action == 0 and direction != "DOWN":
            direction = "UP"
        if action == 1 and direction != "UP":
            direction = "DOWN"
        if action == 2 and direction != "RIGHT":
            direction = "LEFT"
        if action == 3 and direction != "LEFT":
            direction = "RIGHT"
        return direction

    @staticmethod
    def move(direction, squirrel_pos):
        if direction == "UP":
            squirrel_pos[1] -= 10
        if direction == "DOWN":
            squirrel_pos[1] += 10
        if direction == "LEFT":
            squirrel_pos[0] -= 10
        if direction == "RIGHT":
            squirrel_pos[0] += 10
        return squirrel_pos

    def eat(self):
        return self.squirrel_pos[0] == self.food_pos[0] and self.squirrel_pos[1] == self.food_pos[1]

    def spawn_food(self):
        return [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]

    def step(self, action):
        scoreholder = self.score
        reward = 0
        self.direction = SquirrelEnv.change_direction(action, self.direction)
        self.squirrel_pos = SquirrelEnv.move(self.direction, self.squirrel_pos)
        self.squirrel_body.insert(0, list(self.squirrel_pos))
        reward = self.food_handler()
        self.update_game_state()
        reward, done = self.game_over(reward)
        img = self.get_image_array_from_game()   # To get the observations
        info = {'score': self.score}
        self.steps += 1
        time.sleep(self.sleep)
        return img, reward, done, info

    def display_score(self, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    def food_handler(self):
        if self.eat():
            self.score += 1
            reward = 1
            self.food_spawn = False
        else:
            self.squirrel_body.pop()
            reward = 0
        if not self.food_spawn:
            self.food_pos = self.spawn_food()
        self.food_spawn = True
        return reward

    def get_image_array_from_game(self):
        img = pygame.surfarray.array3d(self.game_window)
        img = np.swapaxes(img, 0, 1)
        return img

    def update_game_state(self):
        self.game_window.fill(WHITE)
        for pos in self.squirrel_body:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, BLACK, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

    def game_over(self, reward):
        if self.squirrel_pos[0] < 0 or self.squirrel_pos[0] > self.frame_size_x - 10:
            return -1, True
        if self.squirrel_pos[1] < 0 or self.squirrel_pos[1] > self.frame_size_y - 10:
            return -1, True
        if self.steps >= self.STEP_LIMIT:
            return 0, True
        return reward, False

    def reset(self, seed=None, options=None):
        self.squirrel_pos = [0, 0]
        self.squirrel_body = [[15, 15]]
        self.food_pos = self.spawn_food()
        self.food_spawn = True
        self.direction = "RIGHT"
        self.action = self.direction
        self.score = 0
        self.steps = 0
        img = pygame.surfarray.array3d(self.game_window)
        img = np.swapaxes(img, 0, 1)
        return img

    def render(self, mode='human'):
        if mode == 'human':
            self.game_window.fill(WHITE)  # Clear the screen
            for pos in self.squirrel_body:
                pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 15, 15))
            pygame.draw.circle(self.game_window, BLACK, (self.food_pos[0], self.food_pos[1]), 5)
            pygame.display.flip()

    def close(self):
        pygame.quit()


def render_thread(env):
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        env.render()
        clock.tick(10)  # Adjust the frame rate as needed


env = SquirrelEnv()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(700, 800, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# Start the rendering thread
render_thread_instance = threading.Thread(target=render_thread, args=(env,), daemon=True)
render_thread_instance.start()

# Train the agent
episodes = 1000
update_display_interval = 10

clock = pygame.time.Clock()

for episode in range(episodes):
    state = env.reset()
    state = np.expand_dims(state, axis=0)

    total_reward = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = np.argmax(model.predict(state))
        q_values = model.predict(state)
        print(f"Q-values: {q_values}")
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        target = reward + 0.99 * np.max(model.predict(next_state))
        target_f = model.predict(state)
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0)

        total_reward += reward
        state = next_state

        # Render the environment
        if env.steps % update_display_interval == 0:
            env.render()
            clock.tick(10)  # Adjust the frame rate as needed

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Close the pygame window when training is complete
env.close()

import os
import cv2
import torch
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from ple.games.pixelcopter import Pixelcopter


class Game:
    def __init__(self, game="pixelcopter", fps=30):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.game_name = game
        if game == "flappy":
            engine = FlappyBird()
        elif game == "pixelcopter":
            engine = Pixelcopter()
        else:
            assert False, "This game is not available"
        engine.rewards["loss"] = -5 # reward at terminal state
        self.reward_terminal = -5
        self.game = PLE(engine, fps=fps, display_screen=False)
        self.game.init()
        self.game.act(0) # Start the game by providing arbitrary key as input
        self.key_input = self.game.getActionSet()
        self.reward = 0

    def game_over(self):
        return self.game.game_over()

    def reset_game(self):
        self.game.reset_game()
        self.game.act(0) # Start the game

    def get_image(self):
        return self.game.getScreenRGB()

    def get_torch_image(self):
        image = self.game.getScreenRGB()
        if self.game_name == "flappy":
            image = image[:,:-96,:] # Remove ground
            image = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
            image = np.reshape(image, (84, 84, 1))
        elif self.game_name == "pixelcopter":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.reshape(image, (48, 48, 1))
        image[image > 0] = 1
        image = image.transpose(2, 0, 1) #CHW
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image

    def act(self, action_idx):
        self.reward = self.game.act(self.key_input[action_idx])
        return self.reward


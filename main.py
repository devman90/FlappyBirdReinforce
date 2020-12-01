import os
import sys
import time
import random
import argparse
from glob import glob
from collections import deque

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from game import Game
from model import DQN
from utils import Recorder, init_weights


def train(args):
    model = DQN(game=args.game)
    if args.use_pretrained:
        pretrained_weight = torch.load(
            sorted(glob(os.path.join('ckpt', args.tag, '*.pth')))[-1]
        )
        model.load_state_dict(pretrained_weight)
    else:
        os.makedirs(os.path.join('ckpt', args.tag), exist_ok = True)
        model.apply(init_weights)
    model = model.cuda()
    start = time.time()

    episode = 0
    iteration = 0
    epsilon = args.epsilon

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # instantiate game
    game = Game(game=args.game)
    high_score = 0

    # initialize replay memory
    """
    TO DO

    D =
    """

    elapsed_time = 0
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    score = game.reward
    terminal = game.game_over()

    image_data = game.get_torch_image().cuda()
    state = image_data.unsqueeze(0)

    start = time.time()

    while iteration < args.iteration:
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        # epsilon greedy exploration
        random_action = False
        """
        TO DO

        random_action =
        """

        # Pick action --> random or index of maximum q value
        action_index = 0
        """
        TO DO

        action_index =
        """

        action[action_index] = 1

        elapsed_time = time.time() - start

        # get next state and reward
        reward = game.act(action_index)
        terminal = game.game_over()
        image_data_1 = game.get_torch_image().cuda()

        state_1 = image_data_1.unsqueeze(0)
        action = action.unsqueeze(0).cuda()
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0).cuda()

        # save transition to replay memory
        """
        TO DO
        """

        # if replay memory is full, remove the oldest transition
        """
        TO DO
        """

        # sample random minibatch
        """
        TO DO
        """

        # get output for the next state
        output_1 = model(state_1)

        # Q-learning target
        y = reward if terminal else reward + args.gamma * torch.max(output_1)

        # calculate with target network
        q_value = torch.sum(model(state) * action, dim=1)

        optimizer.zero_grad()
        y = y.detach()
        loss = criterion(q_value, y)

        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1
        score += game.reward

        args.writer.add_scalar('Train/loss', loss, iteration)

        if terminal:
            score = score - game.reward_terminal
            args.writer.add_scalar('Episode/elapsed_time', elapsed_time, episode)
            args.writer.add_scalar('Episode/episode', episode, episode)
            args.writer.add_scalar('Episode/score', score, episode)
            game.reset_game()
            episode += 1
            start = time.time()
            print('Episode {} (Iteration {}): Agent passed {} pipes!, Time: {:.3f}'.format(episode, iteration, score, elapsed_time))
            if score > high_score:
                print('Weight Saved!')
                high_score = score
                torch.save(model,
                           os.path.join('ckpt', args.tag, 'E{:07d}_S{:03d}.pth'.format(episode, int(score)))
                           )
            score = 0
    print("Saving final model")
    torch.save(model,
               os.path.join('ckpt', args.tag, 'E_{:07d}_S{:03d}.pth'.format(episode, int(high_score)))
               )

def test(args):
    model = torch.load(
        sorted(glob(os.path.join('ckpt', args.tag, '*.pth')))[-1],
        map_location='cpu'
    ).eval()
    print('Loaded model: {}'.format(sorted(glob(os.path.join('ckpt', args.tag, '*.pth')))[-1]))
    # initialize video writer
    video_filename = 'output_{}.avi'.format(args.tag)

    dict_screen_shape = {
        "flappy":(288, 512),
        "pixelcopter":(48, 48)}
    out = Recorder(video_filename=video_filename, fps=30,
                   width=dict_screen_shape[args.game][0],
                   height=dict_screen_shape[args.game][1])
    score_list = []
    time_list = []

    game = Game(game=args.game)
    for trials in range(10):

        elapsed_Time = 0
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        terminal = game.game_over()
        start = time.time()
        score = 0

        image_data = game.get_torch_image()
        state = image_data.unsqueeze(0)
        while not terminal:
            output = model(state)[0]
            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            action_index = torch.argmax(output)
            score += game.act(action_index)
            terminal = game.game_over()
            image_data_1 = game.get_torch_image()
            state = image_data_1.unsqueeze(0)

            out.write(game.get_image())

        game.reset_game()
        score = score - game.reward_terminal
        score_list.append(score)
        time_list.append(time.time()-start)
        print('Game Ended!')
        print('Score: {} !'.format(score))

    # Add summary
    out.write_score(sum(score_list), sum(time_list))
    out.save()
    print('Total Score: {}'.format(sum(score_list)))
    print('Total Run Time: {:.3f}'.format(sum(time_list)))


def main(args):
    if args.mode == 'test':
        test(args)

    elif args.mode == 'train':
        train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Q Learning')
    parser.add_argument('--mode', default='train', type=str,
                        help='Mode for network')
    parser.add_argument('--game', default='pixelcopter', type=str,
                        help='{pixelcopter, flappy}')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU Number')
    parser.add_argument('--gamma', default='0.99', type=float,
                        help='Value of gamma')
    parser.add_argument('--epsilon', default='0.02', type=float,
                        help='Value of epsilon')
    parser.add_argument('--iteration', default=1000000, type=int,
                        help='Number of total iterations to run')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--use_pretrained', default=False, type=bool,
                        help='Use pretrained weight')
    parser.add_argument('--tag', default="dqn", type=str,
                        help='name to save')
    parser.add_argument('--writer', default="writer", type=str,
                        help='summarywriter')
    args = parser.parse_args()

    args.writer = SummaryWriter(os.path.join('ckpt', args.tag))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('GPU Enabled: {}'.format(torch.cuda.is_available()))

    main(args)


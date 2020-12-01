# dqn_dsds
Deep Q Learning Challenge for Samsung DSDS AI Course

## How to run

#### Training

> python main.py --tag={some_descriptive_tag} --{some_other_arguments}

#### Test

We automatically load the latest weight trained

> python main.py --mode=test --tag={soome_descriptive_tag}

## Code

* `/assets/`: custom fix for PLE
* `/ple/`: PyGame Learning Environment
* `/ckpt/`: checkpoint and tensorboard logging
* `game.py`: Wrapper for PLE
* `model.py`: DQN
* `utils.py`: miscellaneous functions and classes
* `main.py`: our main learning code

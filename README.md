# Feeding robot

This is Wang Yutong's part2 project for ME5406 in Department of Mechanical Engineering, National University of Singapore

## Prerequisites

see requirement.txt to find the appropriate version of the software package

## Training models

run.py is the main file for training a PR2 robot in static human enviroment using PPO

run_with_human.py is the main file for training a PR2 robot in human cooperation enviroment using PPO

sac.py is the main file for training a PR2 robot in static human enviroment using SAC

sac_cop.py is the main file for training a PR2 robot in human cooperation enviroment using SAC

above file can be used as follows:

```
python run.py --lr 3.0e-4 --gamma 0.99
```

See arguments.py arguments_with_human.py  for a full list of available arguments and hyperparameters.

## Use trained models

Run enjoy_play.py to verify the performance of the trained model in static human enviroment 

Run enjoy_play_with_human.py to verify the performance of the trained model in human cooperation enviroment

```
python enjoy_play.py
```

## Result

summaries\ppo_result is the tensorboard result of ppo in static human enviroment 

summaries\sac_result is the tensorboard result of sac in static human enviroment 

summaries\cop_sac_result  is the tensorboard result of sac in human cooperation enviroment 

summaries\cop_ppo_result  is the tensorboard result of ppo in human cooperation enviroment 

my_video\cooperation is the video of trained model in human cooperation enviroment 

my_video\static_human is the video of trained model in static human enviroment 

## Reference

This library is derived from code baseline:https://github.com/openai/baselines and Assistive Gym:https://github.com/Healthcare-Robotics/assistive-gym

 
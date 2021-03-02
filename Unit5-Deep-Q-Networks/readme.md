
# Deep Q learning using fixed q targets and experience replay

## Results

### Trained Mountain Car :
![](https://media.giphy.com/media/dZopKlQbCgEBTPBy8n/giphy.gif)

### Trained Cart Pole :
![](https://media.giphy.com/media/J5Yh1aY9WhlJc4TZFR/giphy.gif)

## Abstract:

Function approximators like neural networks have succesfully been combined with reinforcement learning because of their ability to derive optimal estimations of the environment using higher order inputs like audios and images.This is an implementation of the Human-level control through deep reinforcement learning with some crunch time tweaks.My implementation was first tested using the low state inputs of the CartPole environment from OpenAi Gym.Then it was succesfully applied to different OpenAi gym  environments without any major hyper-parameter tuning using just the high-dimensional sensory inputs. 


## Environments:

- **CartPole** - [https://gym.openai.com/envs/CartPole-v1/]
- **MountainCar** - [https://gym.openai.com/envs/MountainCar-v0/]

## Instruction:

``` Hyper-parameters tuning for new problems should be done accordingly ```
``` The path to save pytorch model checkpoints should be changed ```

## Dependencies:

- Anaconda: [link](https://docs.anaconda.com/anaconda/install/linux/)
- OpenAi gym: [link](https://gym.openai.com/)
- pytorch: [link](https://pytorch.org/)

## References:

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)


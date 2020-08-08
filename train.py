import os 

import grid2op
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from DQNet_Agent import DQNetAgent
from RandomAgent import RandomAgent
from grid2op.Reward import L2RPNReward


## Hyperparameters
num_frames = 4
batch_size = 32
lr = 1e-5
lr_decay_steps = 100000/4
lr_decay_rate = 0.95
discount_factor = 0.99
tau = 1e-2
buffer_size = 70000
epsilon = 0.99
decay_epsilon = 100000
final_epsilon = 0.01

num_epochs = 10000
num_steps = 1500000
soft_update_freq = -1
hard_update_freq = 1000


env_name = "rte_case14_realistic"

env = grid2op.make(env_name, reward_class=L2RPNReward)

## Wrapper Train Function
def train(env, num_frames, batch_size, buffer_size,
          lr, lr_decay_steps, lr_decay_rate, 
          epsilon, decay_epsilon, final_epsilon,
          discount_factor, tau):
        
        DQAgent = DQNetAgent(env.observation_space,
                             env.action_space,
                             num_frames,
                             batch_size,
                             lr, 
                             lr_decay_steps,
                             lr_decay_rate,
                             discount_factor,
                             tau,
                             buffer_size,
                             epsilon,
                             decay_epsilon,
                             final_epsilon)
                
        epochs, total_steps, losses, avg_losses, net_reward, alive = (
        DQAgent.learn(env, num_epochs, num_steps,
                     soft_update_freq, hard_update_freq))
                     
        return epochs, total_steps, losses, avg_losses, net_reward, alive
## Wrapper Train for RandomAgent
def train_random(env, num_epochs):
    randAgent = RandomAgent(env.observation_space, env.action_space)
    return randAgent.learn(env, num_epochs)


## Main function

# Runs custom agent
epochs, total_steps, losses, avg_losses, net_reward, alive = (
            train(env, num_frames, batch_size, buffer_size, lr, lr_decay_steps,
                    lr_decay_rate, epsilon, decay_epsilon, final_epsilon, 
                    discount_factor, tau))

print("Running Baseline...")
# runs random agent
r_total_steps, r_losses, r_avg_losses, r_net_reward, r_alive = (
    train_random(env, epochs))
    

print('Plotting...')
plt.figure(5)
# survival time for each epoch in custom agent (red line) vs random agent (blue line)
plt.plot(np.arange(epochs), alive, 'r-', np.arange(epochs), r_alive, 'b-')

plt.figure(6)
# total reward in each epoch in custom agent (red line) vs random agent (blue line)
plt.plot(np.arange(epochs), net_reward, 'r-', 
            np.arange(epochs), r_net_reward, 'b-')

plt.figure(7)
# average losses during each epoch 
plt.plot(np.arange(epochs), avg_losses, 'r-')

plt.figure(8)
# total losses during training
plt.plot(np.arange(total_steps), losses, 'r-')
plt.show()


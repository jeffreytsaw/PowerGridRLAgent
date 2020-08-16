import os 
import pickle

import grid2op
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from DQNet_Agent import DQNetAgent
from RandomAgent import RandomAgent
from D3QN_Agent import D3QNetAgent
from grid2op.Reward import L2RPNReward, GameplayReward


## Hyperparameters
num_frames = 4
batch_size = 32

buffer_size = 70000
per_alpha = 0.6
per_beta = 0.4
per_anneal_steps = int(1.5e6)

lr = 1e-5
lr_decay_steps = 100000/4
lr_decay_rate = 0.95

discount_factor = 0.99
tau = 1e-2
lam = 0.7

epsilon = 0.99
decay_epsilon = 100000
final_epsilon = 0.01

num_epochs = 10000
num_steps = int(1.7e6)

soft_update_freq = -1
hard_update_freq = 1000


env_name = "rte_case14_realistic"

env = grid2op.make(env_name, reward_class=L2RPNReward,
                    other_rewards={"gameplay": GameplayReward})

##  Wrapper Train for D3QN
def train_d3qn(env, num_frames, batch_size, 
            buffer_size, per_alpha, per_beta, per_anneal_steps,
            lr, lr_decay_steps, lr_decay_rate, 
            epsilon, decay_epsilon, final_epsilon,
            discount_factor, tau):
                
        D3QAgent = D3QNetAgent(env.observation_space,
                             env.action_space,
                             num_frames,
                             batch_size,
                             lr, 
                             lr_decay_steps,
                             lr_decay_rate,
                             discount_factor,
                             tau,
                             lam,
                             buffer_size,
                             per_alpha,
                             per_beta,
                             per_anneal_steps,
                             epsilon,
                             decay_epsilon,
                             final_epsilon)
        epochs, total_steps, losses, avg_losses, net_reward, alive = (
        D3QAgent.learn(env, num_epochs, num_steps,
                     soft_update_freq, hard_update_freq))
                     
        return epochs, total_steps, losses, avg_losses, net_reward, alive
        
## Wrapper Train for DQN 
def train_dqn(env, num_frames, batch_size, buffer_size,
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
            train_d3qn(env, num_frames, batch_size, buffer_size, 
                    per_alpha, per_beta, per_anneal_steps, 
                    lr, lr_decay_steps, lr_decay_rate, 
                    epsilon, decay_epsilon, final_epsilon, 
                    discount_factor, tau))


print("Running Baseline...")
# runs random agent
'''
r_total_steps, r_losses, r_avg_losses, r_net_reward, r_alive = (
    train_random(env, epochs))
'''

with open('Baseline_data/alive', 'rb') as f:
    r_alive = pickle.load(f)

with open('Baseline_data/net_reward', 'rb') as f:
    r_net_reward = pickle.load(f)
    
#D3 DATA
'''
with open('D3QN_data/alive', 'rb') as f:
    d3_alive = pickle.load(f)

with open('D3QN_data/avg_losses', 'rb') as f:
    d3_avg_losses = pickle.load(f)

with open('D3QN_data/epochs', 'rb') as f:
    d3_epochs = pickle.load(f)

with open('D3QN_data/losses', 'rb') as f:
    d3_losses = pickle.load(f)

with open('D3QN_data/net_reward', 'rb') as f:
    d3_net_reward = pickle.load(f)

with open('D3QN_data/total_steps', 'rb') as f:
    d3_total_steps = pickle.load(f)
'''

#DQ DATA  
with open('DQN_data/alive', 'rb') as f:
    dq_alive = pickle.load(f)

with open('DQN_data/avg_losses', 'rb') as f:
    dq_avg_losses = pickle.load(f)

with open('DQN_data/epochs', 'rb') as f:
    dq_epochs = pickle.load(f)

with open('DQN_data/losses', 'rb') as f:
    dq_losses = pickle.load(f)

with open('DQN_data/net_reward', 'rb') as f:
    dq_net_reward = pickle.load(f)

with open('DQN_data/total_steps', 'rb') as f:
    dq_total_steps = pickle.load(f)

# vector adjustments
epochs = max(dq_epochs, d3_epochs)
while (len(dq_net_reward) < epochs):
    dq_net_reward.append(np.max(np.array(dq_net_reward)))

while (len(dq_alive) < epochs):
    dq_alive.append(np.max(np.array(dq_alive)))

while (len(r_alive) < epochs):
    r_alive.append(np.max(np.array(r_alive)))

while (len(r_net_reward) < epochs):
    r_net_reward.append(np.max(np.array(r_net_reward)))

while (len(dq_avg_losses) < epochs):
    dq_avg_losses.append(np.min(np.array(dq_avg_losses)))



print('Plotting...')
al = plt.figure(5)
# survival time for each epoch in custom agent (red line) vs random agent (blue line)
plt.plot(np.arange(epochs), alive, 'r-', np.arange(epochs), r_alive, 'b-',
        np.arange(epochs), dq_alive, 'g-')
al.suptitle('Survival Duration per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Survival Duration')

re = plt.figure(6)
# total reward in each epoch in custom agent (red line) vs random agent (blue line)
plt.plot(np.arange(epochs), net_reward, 'r-', 
            np.arange(epochs), r_net_reward, 'b-',
            np.arange(epochs), dq_net_reward, 'g-')
re.suptitle('Reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Net Reward')

avl = plt.figure(7)
# average losses during each epoch 
plt.plot(np.arange(epochs), avg_losses, 'r-', 
         np.arange(epochs), dq_avg_losses, 'g-')
avl.suptitle('Average Losses per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')

tl = plt.figure(8)
# total losses during training
plt.plot(np.arange(total_steps), losses, 'r-')
tl.suptitle('Total Loss per Step')
plt.xlabel('Step')
plt.ylabel('Training Loss')

plt.show()

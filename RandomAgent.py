import os

import numpy as np

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

# Agent to perform random actions in environment, used to benchmark approach
class RandomAgent(AgentWithConverter):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)
        
        self.observation_size = self.observation_space.size_obs()
        self.action_size = self.action_space.size()
        
        self.done = False
        
    
    ## Agent Interface
    
    # same as in DQNet_Agent implementation
    def convert_obs(self, obs):
        li_vect=  []
        for el in obs.attr_list_vect:
            v = obs._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)
            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            li_vect.append(v_res)
        return np.concatenate(li_vect)
        
    def convert_act(self, encoded_act):
        return super().convert_act(encoded_act)
    
    def my_act(self, obs):
        return random.randint(0, self.action_size)
        
    ## Training Loop
    def learn(self, env, num_epochs):
        
        epoch = 0
        total_steps = 0
        
        losses = []
        avg_losses = []
        net_reward = []
        alive = []
        
        while (epoch < num_epochs):
            total_reward = 0
            curr_steps = 0
            total_loss = []
            env.reset()
            
            while (not self.done):
                # performs random action
                enc_act = np.random.randint(0, self.action_size)
                
                # steps through environment and collect reward information
                act = self.convert_act(enc_act)
                new_obs, reward, self.done, info = env.step(act)
                
                curr_steps += 1
                total_steps  += 1
                total_reward += reward
                
            alive.append(curr_steps)
            net_reward.append(total_reward) 
            epoch+=1           
            
            
        return (total_steps, losses, avg_losses, net_reward, alive)


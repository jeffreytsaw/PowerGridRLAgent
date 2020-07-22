import os
import math
import numpy as np
import tensorflow as tf


from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from prioritized_replay_buffer import BaseReplayBuffer

from DQNet import DQNet

# Class for a Grid2Op Agent that uses the network defined in DQNet.py to estimate
# Q values

class DQNetAgent(AgentWithConverter):
    def __init__(self, 
                 observation_space,
                 action_space,
                 num_frames = 4,
                 batch_size = 32,
                 learning_rate = 1e-5,
                 learning_rate_decay_steps = 10000,
                 learning_rate_decay_rate = 0.95,
                 discount_factor = 0.95,
                 tau = 1e-2,
                 erb_size = 50000,
                 epsilon = 0.99,
                 decay_epsilon = 1024*32,
                 final_epsilon = 0.0001):
        
        # initializes AgentWithConverter class to handle action conversiions
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)
        
        self.obs_space = observation_space
        self.act_space = action_space
        self.num_frames = num_frames
        
        self.batch_size = batch_size
        self.lr = learning_rate
        self.lr_decay_steps = learning_rate_decay_steps
        self.lr_decay_rate = learning_rate_decay_rate
        self.gamma = discount_factor
        self.tau = tau
        # epsilon is the degree of exploraation
        self.initial_epsilon = epsilon
        # Adaptive epsilon decay constants
        self.decay_epsilon = decay_epsilon
        self.final_epsilon = final_epsilon
        

        self.buff_size = erb_size
        
        self.observation_size = self.obs_space.size_obs()
        self.action_size = self.act_space.size()
        
        self.dqn = DQNet(self.action_size,
                         self.observation_size,
                         self.num_frames,
                         self.lr,
                         self.lr_decay_steps,
                         self.lr_decay_rate,
                         self.batch_size,
                         self.gamma,
                         self.tau)
        
        #State variables
        self.obs = None
        self.done = None
        self.epsilon = self.initial_epsilon
        
        self.state = []
        self.frames = []
        self.next_frames = []
        self.replay_buffer = BaseReplayBuffer(self.buff_size)
        
        return
    
    ## Helper Functions
    
    # Adds to current frame buffer, enforfes length to num_frames
    def update_curr_frame(self, obs):
        self.frames.append(obs.copy())
        if (len(self.frames) > self.num_frames):
            self.frames.pop(0)
        return
    # Adds next frame to next frame buffer, enforces length to num_frames
    def update_next_frame(self, next_obs):
        self.next_frames.append(next_obs.copy())
        if (len(self.next_frames) > self.num_frames):
            self.next_frames.pop(0)
        return
    
    # Adaptive epsilon decay determines next epsilon based off number of steps
    # completed and curren epsilon
    def set_next_epsilon(self, current_step):
        ada_div = self.decay_epsilon / 10.0
        step_off = current_step + ada_div
        ada_eps = self.initial_epsilon * -math.log10((step_off + 1) / 
                                            (self.decay_epsilon + ada_div))
        ada_eps_up_clip = min(self.initial_epsilon, ada_eps)
        ada_eps_low_clip = max(self.final_epsilon, ada_eps_up_clip)
        self.epsilon = ada_eps_low_clip
        return
    
    ## Agent Interface    
    
    #Adapted from l2rpn-baselines from RTE-France
    # Vectorizes observations from grid2op environment for neural network uses
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
    
    # converts encoded action number to action used to interact with grid2op
    # environment
    def convert_act(self, encoded_act):
        return super().convert_act(encoded_act)
    
    # Required for agent evaluation
    # Returns random action or best action as estimated by Q network based on
    # exploration parameter (espilon)
    def my_act(self, state, reward, done):
        if (len(self.frames) < self.num_frames): return 0 #do nothing
        random_act = random.randint(0, self.action_size)
        self.update_curr_frame(state)
        qnet_act, _ = self.dqn.model_action(np.array(self.frames))
        if (np.random.rand(1) < self.epsilon):
            return random_act 
        else:
            return qnet_act

    
    ## Training Loop
    def learn(self,
              env,
              num_epochs,
              num_steps,
              soft_update_freq = 250,
              hard_update_freq = 1000):
        
        #pre-training to fill buffer
        
        print("Starting Pretraining...\n")
        self.done = True
        # Plays random moves and saves the resulting (s, a, r, s', d) pair to 
        # replay buffer. Resets environment when done and continues
        while (len(self.replay_buffer) < self.buff_size):
            if (self.done):
                # reset environment and state parameters
                new_env = env.reset()
                self.frames = []
                self.next_frames = []
                self.done = False
                self.obs = new_env
                self.state = self.convert_obs(self.obs)
                
            self.update_curr_frame(self.state)
           
            # action is random 
            encoded_act = np.random.randint(0, self.action_size)
            act = self.convert_act(encoded_act)
            new_obs, reward, self.done, info = env.step(act)
            
            new_state = self.convert_obs(new_obs)
            self.update_next_frame(new_state)
            
            # only add to buffer if num_frames states are seen
            if (len(self.frames) == self.num_frames and 
                len(self.next_frames) == self.num_frames):
                    
                    agg_state = np.array(self.frames)
                    agg_next_state = np.array(self.next_frames)
                    #(s,a,r,s',d) pair
                    self.replay_buffer.add(agg_state,
                                           encoded_act, reward,
                                           agg_next_state,
                                           self.done)
                    
            self.obs = new_obs
            self.state = new_state
    
        
        epoch = 0 # number of complete runs through environment
        total_steps = 0 # total number of training steps across all epochs
        
        losses = [] # losses[i] is loss from dqn at total_step i
        avg_losses = [] # avg_losses[i] is avg loss during training during epcoh i
        net_reward = [] #net_reward[i] is total reward during epoch i
        alive = [] # alive[i] is number of steps survived for at epoch i
        
        print("Starting training...\n")
        
        # Trains a minimum of num_steps or num_epochs
        while (total_steps < num_steps or epoch < num_epochs):
            
            total_reward = 0
            curr_steps = 0
            total_loss = []
            
            # Reset state parameters
            self.frames = []
            self.next_frames = []
            self.done = False
            self.obs = env.reset()
            self.state = self.convert_obs(self.obs)
            
            # continues until failure 
            while (not self.done):
                
                self.update_curr_frame(self.state)
            
                # Determine action 
                if (len(self.frames) < self.num_frames):
                    enc_act = 0 # do nothing
                elif (np.random.rand(1) < self.epsilon):
                    enc_act = np.random.randint(0, self.action_size)
                else:
                    input = np.array(self.frames)
                    enc_act, _ = self.dqn.model_action(input)
                
                # converts action and steps in environment
                act = self.convert_act(enc_act)
                new_obs, reward, self.done, info = env.step(act)
                new_state = self.convert_obs(new_obs)
                # updates next_state frame
                self.update_next_frame(new_state)
                
                if (len(self.frames) == self.num_frames and
                    len(self.next_frames) == self.num_frames):
                        
                    agg_state = np.array(self.frames)
                    agg_next_state = np.array(self.next_frames)
                    # Adds (s,a,r,s',d) tuple to replay buffer
                    self.replay_buffer.add(agg_state,
                                           encoded_act, reward,
                                           agg_next_state,
                                           self.done)
                # finds the next epsilon 
                self.set_next_epsilon(total_steps)
                
                # samples a batch_size number of experience samples from replay
                # buffer
                s_batch, a_batch, r_batch, s_next_batch, d_batch = (                    
                                    self.replay_buffer.sample(self.batch_size))
               
                # updates network estimates based on replay
                loss = self.dqn.train_on_minibatch(s_batch, a_batch, r_batch, 
                                            s_next_batch, d_batch)
                
                # periodically hard updates the network 
                if (total_steps % hard_update_freq):
                    self.dqn.hard_update_target_network()
                
                # periodically soft update the network    
                elif (total_steps % soft_update_freq):
                    self.dqn.soft_update_target_network()
                
                # update state variables
                self.obs = new_obs
                self.state = new_state
                
                # increase steps, updates metrics
                curr_steps += 1
                total_steps += 1
                total_reward += reward
                losses.append(loss)
                total_loss.append(loss)
            
            # updates metrics throughout epoch
            alive.append(curr_steps)
            net_reward.append(total_reward)            
            avg_losses.append(np.average(np.array(total_loss)))
            
            epoch += 1
            # sanity check to ensure it's working
            if (epoch % 100 == 0):
                print("Completed epoch {}".format(epoch))
                print("Total steps: {}".format(total_steps))
        
        return (epoch, total_steps, losses, avg_losses, net_reward, alive)
            
                
        
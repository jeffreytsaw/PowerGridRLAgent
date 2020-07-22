import os
import numpy as np

import tensorflow as tf

import tensorflow.keras as tfk
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.models as tfkm
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka


# Simple Deep Q Neural Network 
# Features:
#   -Target network for stability. 2 methods of copying network weights, 
#    soft or hard copy.
#   -Leaky ReLu activation for the dense layers
class DQNet(object):
    def __init__(self, 
                action_size, 
                observation_size, 
                num_frames = 4, 
                learning_rate = 1e-5, 
                learning_rate_decay_steps = 10000,
                learning_rate_decay_rate = 0.95,
                minibatch_size = 32,
                discount_factor = 0.95,
                tau = 1e-2):
                
        self.action_size = action_size
        self.observation_size = observation_size
        self.num_frames = num_frames
        self.lr = learning_rate
        # aren't used  decided not to decay the learning rate
        self.lr_decay_steps = learning_rate_decay_steps
        self.lr_decay_rate = learning_rate_decay_rate
        self.batch_size = minibatch_size
        self.discount_factor = discount_factor
        # for soft updates to target network
        self.tau = tau
        
        self.model = None
        self.target_model = None
        
        self.construct_DQN()
        return
      
    # Constructs the Neural Net for Q value estimation
    # 
    # Neural Network Inputs: num_frame observation vectors concatted together
    # Neural Network Outputs: vector of length action_size, where each index 
    #                         represents Q(s,a)
    # Neural Network Hidden Layers: 4 Dense + Leaky ReLU layers that attempt
    #                           to bridge observation_size and action_size
    
    def construct_DQN(self):
        input_layer = tfkl.Input(shape = (self.observation_size * self.num_frames,))
        layer1 = tfkl.Dense(self.observation_size * self.num_frames)(input_layer)
        layer1 = tfka.relu(layer1, alpha = 0.01) #Leaky ReLU
        
        layer2 = tfkl.Dense(self.observation_size * 2)(layer1)
        layer2 = tfka.relu(layer2, alpha = 0.01)
        
        layer3 = tfkl.Dense(self.observation_size)(layer2)
        layer3 = tfka.relu(layer3, alpha = 0.01)
        
        layer4 = tfkl.Dense(self.action_size * 3)(layer3)
        layer4 = tfka.relu(layer4, alpha = 0.01)
        
        layer5 = tfkl.Dense(self.action_size * 2)(layer4)
        layer5 = tfka.relu(layer5, alpha = 0.01)
        
        output = tfkl.Dense(self.action_size)(layer5)
        
        self.model = tfk.Model(inputs = [input_layer],
                               outputs = [output])
        
        self.schedule = tfko.schedules.InverseTimeDecay(self.lr, self.lr_decay_steps, 
                                                        self.lr_decay_rate)
                                                        
        self.optimizer = tfko.Adam(learning_rate=self.schedule)
        
        self.target_model = tfk.Model(inputs = [input_layer],
                                      outputs = [output])
        self.target_model.set_weights(self.model.get_weights())
        return
    
    
    # Trains neural net on a batch sampled from replay buffer
    # s_batch: batch_size number of  full observations (i.e num_frames observations
    #          concatted to a single vector)
    # a_batch: batch_size length list where each index is encoded action taken
    # r_batch: batch_size length list where each index is reward receive for taking
    #           action at a_batch[i]
    # s_next_batch:  batch_size number of full observations of next state
    # d_batch: batch_size length list where each index is True if simulation finished
    #           at that state, false otherwise
    def train_on_minibatch(self, s_batch, a_batch, r_batch, s_next_batch, d_batch): 
        
        # reshaping input vectors
        s_batch = np.reshape(s_batch, (self.batch_size, 
                                       self.observation_size * self.num_frames))
        s_next_batch = np.reshape(s_next_batch, (self.batch_size, 
                                self.observation_size * self.num_frames))
        
                                
        Q_hat = self.model.predict(s_batch, batch_size = self.batch_size)
        
        # Target predictions used for stability
        Q_next_tar = self.target_model.predict(s_next_batch, batch_size = self.batch_size)
        
        # Adjusting new target Q values based on action and reward
        for i in range(self.batch_size):
            Q_hat[i, a_batch[i]] = r_batch[i]
            if (not d_batch[i]):
                Q_hat[i, a_batch[i]] += self.discount_factor * np.max(Q_next_tar[i,:])
        
        # Graadient calculations
        with tf.GradientTape() as tape:
            # tf prediction of actions based on input
            Q_pred = self.model(s_batch) 
            
            #MSE  loss clipped 
            sq_error = tf.math.square(Q_hat-Q_pred)
            batch_error = tf.math.reduce_sum(sq_error, axis=1)
            batch_loss = tf.clip_by_value(batch_error, 0, 1e4)
            #calculating batch loss
            loss = tf.math.reduce_mean(batch_loss)
         
        # propogate gradients through network
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
        return loss.numpy()
    
    # Outputs action with highest Q value based on model estimataes and input
    def model_action(self, input):
        input = input.reshape(1, self.observation_size * self.num_frames)
        Q_pred = self.model.predict(input, batch_size = 1)
        # returns highest Q value action number, and action_list
        return (np.argmax(Q_pred), Q_pred[0])
    
    # Soft updates target network loss by weighting new weights based on current
    # target network weights and model weights
    def soft_update_target_network(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            # new weights are weighted sum of model weights and target weights
            # weighted by tau factor
            model_weights[i] = model_weights[i]*self.tau + target_weights[i]*(1 - self.tau)
        self.target_model.set_weights(model_weights)
        return
    
    # Sets target model weights to model weights
    def hard_update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        return
        
        
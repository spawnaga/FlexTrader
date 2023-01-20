# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:51:50 2023

@author: alial
"""
import os
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Lambda
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import pickle


# Start the asyncio event loop


def policy_gradient_loss(y_true, y_pred):
    # Define the loss function
    action_prob = y_pred[:, :-1]
    advantages = y_pred[:, -1]

    cross_entropy = tf.keras.backend.categorical_crossentropy(y_true, action_prob)
    loss = tf.reduce_mean(cross_entropy * advantages)
    return loss


class Memory:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.priorities = []

    def add(self, experience, priority=1):
        if len(self.buffer) == self.max_size:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]
        indices = random.choices(range(len(self.buffer)), k=batch_size, weights=probs)
        experiences = [self.buffer[i] for i in indices]
        return experiences

    def _size(self):
        return len(self.buffer)

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p

    def clear(self):
        self.buffer = []


class MultiTask:
    def __init__(self, task, state, state_size, action_size, num_outputs_1=5, num_outputs_2=5):
        self.state = state
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        if task == 'dqn':
            # Initialize DQN model
            self.dqn_model = self._build_dqn_model(num_outputs_1, num_outputs_2)
            self.dqn_memory = Memory(max_size=50000)
            self.dqn_gamma = 0.95
            self.dqn_epsilon = 1.0
            self.dqn_epsilon_min = 0.01
            self.dqn_epsilon_decay = 0.995
            self.dqn_learning_rate = 0.001
        elif task == 'ddqn':
            # Initialize DDQN model
            self.ddqn_model = self._build_ddqn_model(num_outputs_1, num_outputs_2)
            self.ddqn_target_model = self._build_ddqn_model(num_outputs_1, num_outputs_2)
            self.ddqn_memory = Memory(max_size=50000)
            self.ddqn_gamma = 0.95
            self.ddqn_epsilon = 1.0
            self.ddqn_epsilon_min = 0.01
            self.ddqn_epsilon_decay = 0.995
            self.ddqn_learning_rate = 0.001
        elif task == 'actor_critic':
            # Initialize actor-critic model
            self.actor_critic_model = self._build_actor_critic_model(num_outputs_1, num_outputs_2)
            self.actor_critic_memory = Memory(max_size=50000)
            self.actor_critic_gamma = 0.95
            self.actor_critic_alpha = 0.001
            self.actor_critic_alpha_decay = 0.995
            self.actor_critic_alpha_min = 0.01
            self.actor_critic_epsilon = 1.0
            self.actor_critic_epsilon_decay = 0.995
            self.actor_critic_epsilon_min = 0.01
        elif task == 'policy_gradient':
            # Initialize PPO model
            self.policy_gradient_learning_rate = 0.001
            self.policy_gradient_model = self._build_policy_gradient_model(num_outputs_1, num_outputs_2)
            self.policy_gradient_memory = Memory(max_size=50000)
            self.policy_gradient_gamma = 0.95
            self.policy_gradient_epsilon = 1.0
            self.policy_gradient_alpha_decay = 0.995
            self.policy_gradient_alpha_min = 0.01

    def _build_dqn_model(self, num_outputs_1, num_outputs_2):
        # Define the input layer
        input_layer = Input(shape=(self.state_size,))
        x = input_layer

        # Define the hidden layers using CuDNNLSTM
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = CuDNNLSTM(64, return_sequences=True)(x)
        x = CuDNNLSTM(32, return_sequences=True)(x)
        x = CuDNNLSTM(16, return_sequences=True)(x)
        x = CuDNNLSTM(8)(x)

        # Define the output layers for the first and second tasks
        output = Dense(num_outputs_2, activation='linear')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def _build_ddqn_model(self, num_outputs_1, num_outputs_2):
        # Define the input layer
        input_layer = Input((self.state_size,))
        x = input_layer

        # Define the hidden layers using CuDNNLSTM
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = CuDNNLSTM(64, return_sequences=True)(x)
        x = CuDNNLSTM(32, return_sequences=True)(x)
        x = CuDNNLSTM(16, return_sequences=True)(x)
        x = CuDNNLSTM(8)(x)

        # Define the output layers for the first and second tasks
        output = Dense(num_outputs_2, activation='linear')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def _build_actor_critic_model(self, num_outputs_1, num_outputs_2):
        # Define the input layer
        input_layer = Input((self.state_size,))
        x = input_layer

        # Define the hidden layers using CuDNNLSTM
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = CuDNNLSTM(64, return_sequences=True)(x)
        x = CuDNNLSTM(32, return_sequences=True)(x)
        x = CuDNNLSTM(16, return_sequences=True)(x)
        x = CuDNNLSTM(8)(x)

        # Define the output layers for the actor and critic
        critic_output = Dense(num_outputs_1, activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=critic_output)

        # Compile the model using the Adam optimizer
        model.compile(optimizer=Adam(0.001), loss=['categorical_crossentropy', 'mse'])

        return model

    def _build_policy_gradient_model(self, num_outputs_1, num_outputs_2):
        # Define the input layer
        input_layer = Input((self.state_size,))
        x = input_layer

        # Define the hidden layers using CuDNNLSTM
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = CuDNNLSTM(64, return_sequences=True)(x)
        x = CuDNNLSTM(32, return_sequences=True)(x)
        x = CuDNNLSTM(16, return_sequences=True)(x)
        x = CuDNNLSTM(8)(x)

        # Define the output layers for the first and second tasks
        output = Dense(num_outputs_2, activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def add_dqn_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the DQN task
        transition = (state, action, reward, next_state, done)
        self.dqn_memory.add(transition)

    def add_ddqn_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the DDQN task
        transition = (state, action, reward, next_state, done)
        self.ddqn_memory.add(transition)

    def add_actor_critic_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the actor-critic task
        transition = (state, action, reward, next_state, done)
        self.actor_critic_memory.add(transition)

    def add_policy_gradient_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the policy gradient task
        transition = (state, action, reward, next_state, done)
        self.policy_gradient_memory.add(transition)

    def replay_dqn(self, batch_size):
        experiences = self.dqn_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.concatenate(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.concatenate(next_states)
        target_Qs = self.dqn_model.predict(states)
        target_Qs_next = self.dqn_model.predict(next_states)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        actions = actions.reshape(-1, 1)

        target_Qs[np.arange(batch_size), actions] = rewards + self.dqn_gamma * np.max(target_Qs_next, axis=1) * (
                    1 - dones)

        self.dqn_model.fit(states, target_Qs, epochs=1, verbose=0)
        self.dqn_epsilon *= self.dqn_epsilon_decay
        self.dqn_epsilon = max(self.dqn_epsilon_min, self.dqn_epsilon)

        # print("000000000")

    def replay_ddqn(self, batch_size):
        experiences = self.ddqn_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)

        q_values = self.ddqn_model.predict(states)
        next_q_values = self.ddqn_target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(self.ddqn_model.predict(next_states[i:i + 1]), axis=1)
                q_values[i][actions[i]] = rewards[i] + self.ddqn_gamma * next_q_values[i][a]

        self.ddqn_model.fit(states, q_values, verbose=0)
        # print("111111111")

    def replay_actor_critic(self, batch_size):
        """Method for training the actor-critic model using experience replay"""
        # Sample a batch of experiences from the memory
        experiences = self.actor_critic_memory.sample(batch_size)
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
    
        # Predict the Q-values of the next states

        next_q_values = self.actor_critic_model.predict(next_states.squeeze())

        # Compute the target Q-values
        target_q_values = rewards + self.actor_critic_gamma * np.amax(next_q_values, axis=1) * (1 - dones)
    
        # Update the Q-values of the current states
        target_q_values_batch = self.actor_critic_model.predict(states.squeeze())
        for i, action in enumerate(actions):
            target_q_values_batch[i, action] = target_q_values[i]
    
        # Fit the model on the experiences
        self.actor_critic_model.fit(states.squeeze(), target_q_values_batch, epochs=1, verbose=0)
        self.actor_critic_alpha *= self.actor_critic_alpha_decay
        self.actor_critic_alpha = max(self.actor_critic_alpha_min, self.actor_critic_alpha)
        self.actor_critic_model.optimizer.learning_rate = self.actor_critic_alpha


    def replay_policy_gradient(self, batch_size):
        """Replay the experiences from the memory for the policy gradient task"""
        experiences = self.policy_gradient_memory.sample(batch_size)
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
    
        # Predict the Q-values of the next states
        next_q_values = self.policy_gradient_model.predict(next_states.squeeze())
        # Compute the target Q-values
        target_q_values = rewards + self.policy_gradient_gamma * np.amax(next_q_values, axis=1) * (1 - dones)
    
        # Update the Q-values of the current states
        target_q_values_batch = self.policy_gradient_model.predict(states.squeeze())
        for i, action in enumerate(actions):
            target_q_values_batch[i, action] = target_q_values[i]
    
        # Fit the model on the experiences
        self.policy_gradient_model.fit(states.squeeze(), target_q_values_batch, epochs=1, verbose=0)

    def act(self, state, task, job='test'):
        """Method for getting the next action for the agent to take"""
        state = self.normalize_data(state)
    
        if task == 'dqn':
            if job == 'train' and np.random.rand() <= self.dqn_epsilon:
                print('random')
                return random.randrange(self.action_size)
            q_values = self.dqn_model.predict(state)
            return np.argmax(q_values[0])
    
        elif task == 'ddqn':
            if job == 'train' and np.random.rand() <= self.ddqn_epsilon:
                print('random')
                return random.randrange(self.action_size)
            q_values = self.ddqn_model.predict(state)
            return np.argmax(q_values[0])

        if task == 'actor_critic':
            if np.random.rand() <= self.actor_critic_epsilon and job == 'train':
                print('random')
                return random.randrange(self.action_size)
            probs = self.actor_critic_model.predict(state)
            action = np.argmax(probs[0])
            self.actor_critic_epsilon = max(self.actor_critic_epsilon * self.actor_critic_epsilon_decay, self.actor_critic_epsilon_min)
            return action

        elif task == 'policy_gradient':
            if job == 'train':
                self.policy_gradient_epsilon *= self.policy_gradient_alpha_decay
                self.policy_gradient_epsilon = max(self.policy_gradient_epsilon, self.policy_gradient_alpha_min)
                if np.random.rand() <= self.policy_gradient_epsilon:
                    print('random')
                    return random.randrange(self.action_size)
            probs = self.policy_gradient_model.predict(state)[0]
            return np.argmax(probs)



    def load(self, name, task, folder_name='model1'):
        if task == 'dqn':
            self.dqn_model.load_weights(os.path.join(folder_name, f'{name}_dqn.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                self.dqn_memory = pickle.load(f)
        elif task == 'ddqn':
            self.ddqn_model.load_weights(os.path.join(folder_name, f'{name}_ddqn.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                self.ddqn_memory = pickle.load(f)
        elif task == 'actor_critic':
            self.actor_critic_model.load_weights(os.path.join(folder_name, f'{name}_actor_critic.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                self.actor_critic_memory = pickle.load(f)
        elif task == 'policy_gradient':
            self.policy_gradient_model.load_weights(os.path.join(folder_name, f'{name}_policy_gradient.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                self.policy_gradient_memory = pickle.load(f)

    def save(self, name, task, folder_name='model1'):
        # Check if the folder already exists
        if not os.path.exists(folder_name):
            # Create the folder if it does not exist
            os.mkdir(folder_name)
        if task == 'dqn':
            self.dqn_model.save_weights(os.path.join(folder_name, f'{name}_dqn.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.dqn_memory, f)
        elif task == 'ddqn':
            self.ddqn_model.save_weights(os.path.join(folder_name, f'{name}_ddqn.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.ddqn_memory, f)
        elif task == 'actor_critic':
            self.actor_critic_model.save_weights(os.path.join(folder_name, f'{name}_actor_critic.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.actor_critic_memory, f)
        elif task == 'policy_gradient':
            self.policy_gradient_model.save_weights(os.path.join(folder_name, f'{name}_policy_gradient.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.policy_gradient_memory, f)

    def normalize_data(self, data):
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Fit the scaler to the data
        scaler.fit(data)

        # Transform the data
        normalized_data = scaler.transform(data)

        return normalized_data

    def calculate_reward(self, action, market, i, row, previous_row, slippage=0.05, transaction_cost=0.25):
        """Method for calculating the reward for a given action"""

        # Get the current price and previous price
        current_price = row['close']
        previous_price = previous_row['close']

        # Initialize the reward
        reward = 0

        # Check if the action is to buy
        if action == 0:
            # Calculate the reward based on the current and previous prices, slippage, and transaction cost
            reward = (current_price - previous_price) - (current_price * slippage) - transaction_cost
        # Check if the action is to sell
        elif action == 1:
            # Calculate the reward based on the current and previous prices, slippage, and transaction cost
            reward = (previous_price - current_price) - (current_price * slippage) - transaction_cost
        # Check if the action is to hold
        elif action == 2:
            # Calculate the reward based on the current and previous prices
            reward = current_price - previous_price
        return reward

    def incorporate_other_data(self, other_data):
        """
        Incorporates other relevant information into the state, such as technical indicators or fundamental data.
        :param other_data: Numpy array containing the other relevant data.
        """
        self.state = np.concatenate((self.state, other_data), axis=1)

    def add_to_memory(self, task, state, action, reward, next_state, done):
        # Add the transition to the memory for the specified task
        if task == 'dqn':
            self.add_dqn_transition(state, action, reward, next_state, done)
        if task == 'ddqn':
            self.add_ddqn_transition(state, action, reward, next_state, done)
        elif task == 'actor_critic':
            self.add_actor_critic_transition(state, action, reward, next_state, done)
        elif task == 'policy_gradient':
            self.add_policy_gradient_transition(state, action, reward, next_state, done)

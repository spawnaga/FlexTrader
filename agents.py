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
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import pickle
import sys


# Start the asyncio event loop

class Memory:
    def __init__(self, task=None, max_size=100):
        # if not hasattr(self, task):
        #     return
        # else:
        self.buffer = []
        if not hasattr(self, 'priorities'):
            self.priorities = []
        else:
            if self.priorities is None:
                self.priorities = []
        self.max_size = max_size

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
    def __init__(self, task, state, state_size, action_size, job='test', layers=5):
        self.state = state
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        if job == 'train':
            self.epsilon = 1.0
        else:
            self.epsilon = -1
        if task == 'dqn':
            # Initialize DQN model
            self.dqn_model = self._build_model(action_size, layers)
            self.dqn_memory = Memory(task, max_size=50000)
            self.dqn_gamma = 0.95
            self.dqn_epsilon = self.epsilon
            self.dqn_epsilon_min = 0.2
            self.dqn_epsilon_decay = 0.995
            self.dqn_learning_rate = 0.001
        elif task == 'ddqn':
            # Initialize DDQN model
            self.ddqn_model = self._build_model(action_size, layers)
            self.ddqn_target_model = self._build_model(action_size, layers)
            self.ddqn_memory = Memory(task, max_size=50000)
            self.ddqn_gamma = 0.95
            self.ddqn_epsilon = self.epsilon
            self.ddqn_epsilon_min = 0.2
            self.ddqn_epsilon_decay = 0.995
            self.ddqn_learning_rate = 0.001
        elif task == 'actor_critic':
            # Initialize actor-critic model
            self.actor_critic_model = self._build_model(action_size, layers)
            self.actor_critic_memory = Memory(task, max_size=50000)
            self.actor_critic_gamma = 0.95
            self.actor_critic_alpha = 0.001
            self.actor_critic_alpha_decay = 0.995
            self.actor_critic_alpha_min = 0.01
            self.actor_critic_epsilon = self.epsilon
            self.actor_critic_epsilon_decay = 0.995
            self.actor_critic_epsilon_min = 0.2
        elif task == 'policy_gradient':
            # Initialize PPO model
            self.policy_gradient_learning_rate = 0.01
            self.policy_gradient_model = self._build_model(action_size, layers)
            self.policy_gradient_memory = Memory(task, max_size=50000)
            self.policy_gradient_gamma = 0.95
            self.policy_gradient_epsilon = self.epsilon
            self.policy_gradient_alpha_decay = 0.995
            self.policy_gradient_alpha_min = 0.2

    def _build_model(self, num_outputs, layers=2):
        # Define the input layer
        input_layer = Input((self.state_size,))
        x = input_layer

        # Define the hidden layers using CuDNNLSTM
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        for _ in range(layers):
            x = CuDNNLSTM(100, return_sequences=True)(x)  # number of hidden layers

        # Define the output layers for the first and second tasks
        output = Dense(num_outputs, activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy')

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
        if self.dqn_memory._size() < batch_size:
            return
        experiences = self.dqn_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.concatenate(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.concatenate(next_states)
        target_Qs = self.dqn_model.predict(states.squeeze()).squeeze()
        target_Qs_next = self.dqn_model.predict(next_states.squeeze()).squeeze()
        actions = np.reshape(actions, (batch_size,))
        dones = np.reshape(dones, (batch_size,))
        rewards = np.reshape(rewards, (batch_size,))

        target_Qs[np.arange(batch_size), actions] = rewards + self.dqn_gamma * np.max(target_Qs_next, axis=1) * (
                1 - dones)
        self.dqn_model.fit(states, np.expand_dims(target_Qs, axis=1), epochs=1, verbose=0)

    def replay_ddqn(self, batch_size):
        if self.ddqn_memory._size() < batch_size:
            return
        experiences = self.ddqn_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)
        actions = np.reshape(actions, (batch_size,))
        dones = np.reshape(dones, (batch_size,))
        rewards = np.reshape(rewards, (batch_size,))

        q_values = self.ddqn_model.predict(states.squeeze()).squeeze()
        next_q_values = self.ddqn_target_model.predict(next_states.squeeze()).squeeze()

        a = np.argmax(self.ddqn_model.predict(next_states.squeeze()).squeeze(), axis=1)
        q_values[np.arange(batch_size), actions] = rewards + self.ddqn_gamma * next_q_values[
            np.arange(batch_size), a] * (1 - dones)

        self.ddqn_model.fit(states, np.expand_dims(a=q_values, axis=1), verbose=0)
        self.update_target_model()

    def update_target_model(self):
        self.ddqn_target_model.set_weights(self.ddqn_model.get_weights())

    def replay_actor_critic(self, batch_size):
        """Method for training the actor-critic model using experience replay"""
        # Sample a batch of experiences from the memory
        if self.actor_critic_memory._size() < batch_size:
            return
        experiences = self.actor_critic_memory.sample(batch_size)
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])

        # Predict the Q-values of the next states
        next_q_values = self.actor_critic_model.predict(next_states.squeeze()).squeeze()

        # Compute the target Q-values
        target_q_values = rewards + self.actor_critic_gamma * (1 - dones) * np.amax(next_q_values, axis=1)

        # Update the Q-values of the current states
        target_q_values_batch = self.actor_critic_model.predict(states.squeeze()).squeeze()
        for i, action in enumerate(actions):
            target_q_values_batch[i, action] = target_q_values[i]

        # Fit the model on the experiences
        self.actor_critic_model.fit(states.squeeze(), np.expand_dims(target_q_values_batch, axis=1), epochs=1,
                                    verbose=0)

        self.actor_critic_alpha *= self.actor_critic_alpha_decay
        self.actor_critic_alpha = max(self.actor_critic_alpha_min, self.actor_critic_alpha)
        self.actor_critic_model.optimizer.learning_rate = self.actor_critic_alpha

    def replay_policy_gradient(self, batch_size):
        experiences = self.policy_gradient_memory.sample(batch_size)
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])

        # Compute the advantages
        advantages = rewards - np.mean(rewards)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # Create a GradientTape to record the operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass of the model
            logits = self.policy_gradient_model(states.squeeze())
            logits = tf.reshape(logits, (-1, 5))
            log_probs = tf.nn.log_softmax(logits)

            # Sample actions from the policy
            actions = tf.random.categorical(log_probs, 1)

            # Compute the negative log likelihood of the actions taken
            negative_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf.one_hot(actions, 5))

            # Compute the loss as the mean of the negative log likelihood
            loss = tf.reduce_mean(negative_log_likelihood * advantages)
            # print('policy_gradient loss is', loss.numpy().item())
        # Compute the gradients of the loss with respect to the model's trainable weights
        grads = tape.gradient(loss, self.policy_gradient_model.trainable_weights)
        # Apply the gradients to the model's optimizer
        self.policy_gradient_model.optimizer.apply_gradients(zip(grads, self.policy_gradient_model.trainable_weights))

    # @tf.function(experimental_relax_shapes=True)
    def act(self, state, task, job='test'):
        """Method for getting the next action for the agent to take"""
        # state = self.normalize_data(state)

        if task == 'dqn':
            if job == 'train' and np.random.rand() <= self.dqn_epsilon:
                self.dqn_epsilon *= self.dqn_epsilon_decay
                self.dqn_epsilon = max(self.dqn_epsilon, self.dqn_epsilon_min)
                return random.randrange(self.action_size)
            q_values = self.dqn_model.predict(state).reshape(-1)
            return np.argmax(q_values)

        elif task == 'ddqn':
            if job == 'train' and np.random.rand() <= self.ddqn_epsilon:
                self.ddqn_epsilon *= self.ddqn_epsilon_decay
                self.ddqn_epsilon = max(self.ddqn_epsilon, self.ddqn_epsilon_min)
                return random.randrange(self.action_size)
            q_values = self.ddqn_model.predict(state).reshape(-1)
            return np.argmax(q_values)

        if task == 'actor_critic':
            if job == 'train':
                self.actor_critic_epsilon *= self.actor_critic_epsilon_decay
                self.actor_critic_epsilon = max(self.actor_critic_epsilon, self.actor_critic_epsilon_min)
                if np.random.rand() <= self.actor_critic_epsilon:
                    return random.randrange(self.action_size)
            probs = self.actor_critic_model.predict(state).reshape(-1)
            return np.argmax(probs)

        elif task == 'policy_gradient':
            if job == 'train':
                self.policy_gradient_epsilon *= self.policy_gradient_alpha_decay
                self.policy_gradient_epsilon = max(self.policy_gradient_epsilon, self.policy_gradient_alpha_min)
                if np.random.rand() <= self.policy_gradient_epsilon:
                    return random.randrange(self.action_size)
            probs = self.policy_gradient_model.predict(state).reshape(-1)
            return np.argmax(probs)

    def load(self, name, task, folder_name='model1'):
        if os.path.exists(folder_name):
            if task == 'dqn' and os.path.exists(f'{name}_ddqn.h5'):
                self.dqn_model.load_weights(os.path.join(folder_name, f'{name}_dqn.h5'))
                with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                    self.dqn_memory = pickle.load(f)
            elif task == 'ddqn' and os.path.exists(f'{name}_dqn.h5'):
                self.ddqn_model.load_weights(os.path.join(folder_name, f'{name}_ddqn.h5'))
                with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                    self.ddqn_memory = pickle.load(f)
            elif task == 'actor_critic' and os.path.exists(f'{name}_actor_critic.h5'):
                self.actor_critic_model.load_weights(os.path.join(folder_name, f'{name}_actor_critic.h5'))
                with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                    self.actor_critic_memory = pickle.load(f)
            elif task == 'policy_gradient' and os.path.exists(f'{name}_policy_gradient.h5'):
                self.policy_gradient_model.load_weights(os.path.join(folder_name, f'{name}_policy_gradient.h5'))
                with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                    self.policy_gradient_memory = pickle.load(f)
        else:
            return

    def save(self, name, task, folder_name='model1'):
        # Check if the folder already exists
        if not os.path.exists(folder_name):
            # Create the folder if it does not exist
            os.mkdir(folder_name)
        if task == 'dqn':
            self.dqn_model.save_weights(os.path.join(folder_name, f'{name}_dqn.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.dqn_memory, f)
            # Verify if saving memory went right and the file is not corrupted
            corrupted = self.verify_pickle(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), self.dqn_memory)
            if corrupted:
                self.save(name, task)
        elif task == 'ddqn':
            self.ddqn_model.save_weights(os.path.join(folder_name, f'{name}_ddqn.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.ddqn_memory, f)
            # Verify if saving memory went right and the file is not corrupted
            corrupted = self.verify_pickle(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), self.ddqn_memory)
            if corrupted:
                self.save(name, task)
        elif task == 'actor_critic':
            self.actor_critic_model.save_weights(os.path.join(folder_name, f'{name}_actor_critic.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.actor_critic_memory, f)
            # Verify if saving memory went right and the file is not corrupted
            corrupted = self.verify_pickle(os.path.join(folder_name, f'{name}_{task}_memory.pickle'),
                                           self.actor_critic_memory)
            if corrupted:
                self.save(name, task)
        elif task == 'policy_gradient':
            self.policy_gradient_model.save_weights(os.path.join(folder_name, f'{name}_policy_gradient.h5'))
            with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'wb') as f:
                pickle.dump(self.policy_gradient_memory, f)
            # Verify if saving memory went right and the file is not corrupted
            corrupted = self.verify_pickle(os.path.join(folder_name, f'{name}_{task}_memory.pickle'),
                                           self.policy_gradient_model)
            if corrupted:
                self.save(name, task)

    def check_pickle_size(self, pickle_file, original_data):
        # Use to check if trained model is bigger that the trained model
        pickle_size = os.path.getsize(pickle_file)
        original_size = sys.getsizeof(original_data)
        if pickle_size < original_size:
            return True
        else:
            return False

    def verify_pickle(self, pickle_file, original_data):
        # Use to verify if saving pickle file went fine
        pickle_size = os.path.getsize(pickle_file)
        original_size = sys.getsizeof(original_data)

        if pickle_size < original_size:
            return True
        else:
            return False

    def normalize_data(self, data):
        # Create a StandardScaler object
        if data is None or len(data) == 0:
            return 2
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

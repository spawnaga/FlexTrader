# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:51:50 2023

@author: alial
"""
import datetime
import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Memory:
    def __init__(self, max_size=100):
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


def normalize_data(data):
    # Create a StandardScaler object
    if data is None or len(data) == 0:
        return 2
    scaler = StandardScaler()

    # Fit the scaler to the data
    scaler.fit(data)

    # Transform the data
    normalized_data = scaler.transform(data)

    return normalized_data


def check_pickle_size(pickle_file, original_data):
    # Use to check if trained model is bigger that the trained model
    pickle_size = os.path.getsize(pickle_file)
    original_size = sys.getsizeof(original_data)
    if pickle_size < original_size:
        return True
    else:
        return False


def verify_pickle(pickle_file, original_data):
    # Use to verify if saving pickle file went fine
    pickle_size = os.path.getsize(pickle_file)
    original_size = sys.getsizeof(original_data)

    if pickle_size < original_size:
        return True
    else:
        return False


def calculate_reward(action, row, previous_row, slippage=0.05, transaction_cost=0.25):
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


class MultiTask:
    def __init__(self, state_size, action_size=5, layers=3):
        self.q_values = []
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.01
        self.temperature = 1.0
        self.tasks = {0: 'dqn',
                      1: 'ddqn',
                      2: 'actor_critic',
                      3: 'policy_gradient',
                      }

        self.master_model, self.master_log = self._build_model(len(self.tasks.keys()), 'master', layers)
        self.master_memory = Memory()
        self.master_gamma = 0.95
        self.master_learning_rate = 0.001
        self.master_performance = 1.0
        self.master_performance_count = 0

        # Initialize DQN model
        self.dqn_learning_rate = 0.1
        self.dqn_model, self.dqn_log = self._build_model(action_size, 'dqn', layers)
        self.dqn_memory = Memory()
        self.dqn_gamma = 0.95
        self.dqn_epsilon = 3
        self.dqn_epsilon_min = 0.6
        self.dqn_epsilon_decay = 0.95
        self.dqn_learning_rate = 0.001

        # Initialize DDQN model
        self.ddqn_learning_rate = 0.1
        self.ddqn_model, self.ddqn_log = self._build_model(action_size, 'ddqn', layers)
        self.ddqn_target_model, self.ddqn_target_log = self._build_model(action_size, 'ddqn_target', layers)
        self.ddqn_memory = Memory()
        self.ddqn_gamma = 0.95
        self.ddqn_epsilon = 3
        self.ddqn_epsilon_min = 0.4
        self.ddqn_epsilon_decay = 0.995

        # Initialize actor-critic model
        self.actor_critic_learning_rate = 0.1
        self.actor_critic_model, self.actor_critic_log = self._build_model(action_size, 'actor_critic', layers)
        self.actor_critic_memory = Memory()
        self.actor_critic_gamma = 0.95
        self.actor_critic_alpha = 0.001
        self.actor_critic_alpha_decay = 0.995
        self.actor_critic_alpha_min = 0.01
        self.actor_critic_epsilon = 3
        self.actor_critic_epsilon_decay = 0.995
        self.actor_critic_epsilon_min = 0.4

        # Initialize PPO model
        self.policy_gradient_learning_rate = 0.1
        self.policy_gradient_model, self.policy_gradient_log = self._build_model(action_size, "policy_gradient",
                                                                                 layers=6)
        self.policy_gradient_memory = Memory()
        self.policy_gradient_gamma = 0.95
        self.policy_gradient_epsilon = 3
        self.policy_gradient_alpha_decay = 0.995
        self.policy_gradient_alpha_min = 0.4

    def _build_model(self, num_outputs, task, layers=3):
        # Define the input layer
        input_layer = Input((self.state_size,))
        x = input_layer
        logdir = "logs/fit/" + task + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Define the hidden layers using CuDNNLSTM
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        for _ in range(layers):
            x = CuDNNLSTM(64, return_sequences=True)(x)  # number of hidden layers

        # Define the output layers for the first and second tasks
        output = Dense(num_outputs, activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Learning rate scheduler
        boundaries = [100, 200, 500, 1000, 3000]
        values = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy')

        return model, logdir

    def add_master_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the master task
        transition = (state, action, reward, next_state, done)
        self.master_memory.add(transition)

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

    def replay_master(self, batch_size):

        experiences = self.master_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.concatenate(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.concatenate(next_states)
        target_Qs = self.master_model.predict(states.squeeze()).squeeze()
        target_Qs_next = self.master_model.predict(next_states.squeeze()).squeeze()
        actions = np.reshape(actions, (batch_size,))
        dones = np.reshape(dones, (batch_size,))
        rewards = np.reshape(rewards, (batch_size,))
        target_Qs[np.arange(batch_size), actions] = rewards + self.master_gamma * np.max(target_Qs_next, axis=1) * (
                1 - dones)

        with tf.summary.create_file_writer(self.master_log).as_default():
            history = self.master_model.fit(states, np.expand_dims(target_Qs, axis=1), epochs=1, verbose=0)
            tf.summary.scalar("loss", history.history['loss'][0], step=self.master_model.optimizer.iterations)

    def replay_dqn(self, batch_size):

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
        with tf.summary.create_file_writer(self.dqn_log).as_default():
            history = self.dqn_model.fit(states, np.expand_dims(target_Qs, axis=1), epochs=1, verbose=0)
            tf.summary.scalar("loss", history.history['loss'][0], step=self.dqn_model.optimizer.iterations)

    def replay_ddqn(self, batch_size):

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
        with tf.summary.create_file_writer(self.ddqn_log).as_default():
            history = self.ddqn_model.fit(states, np.expand_dims(a=q_values, axis=1), verbose=0)
            tf.summary.scalar("loss", history.history['loss'][0], step=self.ddqn_model.optimizer.iterations)
        self.update_target_model()

    def update_target_model(self):
        self.ddqn_target_model.set_weights(self.ddqn_model.get_weights())

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
        next_q_values = self.actor_critic_model.predict(next_states.squeeze()).squeeze()

        # Compute the target Q-values
        target_q_values = rewards + self.actor_critic_gamma * (1 - dones) * np.amax(next_q_values, axis=1)

        # Update the Q-values of the current states
        target_q_values_batch = self.actor_critic_model.predict(states.squeeze()).squeeze()
        for i, action in enumerate(actions):
            target_q_values_batch[i, action] = target_q_values[i]

        # Fit the model on the experiences
        with tf.summary.create_file_writer(self.actor_critic_log).as_default():
            history = self.actor_critic_model.fit(states.squeeze(), np.expand_dims(target_q_values_batch, axis=1),
                                                  epochs=1,
                                                  verbose=0)
            tf.summary.scalar("loss", history.history['loss'][0], step=self.actor_critic_model.optimizer.iterations)
        self.actor_critic_alpha *= self.actor_critic_alpha_decay
        self.actor_critic_alpha = max(self.actor_critic_alpha_min, self.actor_critic_alpha)
        self.actor_critic_model.optimizer.learning_rate = self.actor_critic_alpha

    def replay_policy_gradient(self, batch_size):

        experiences = self.policy_gradient_memory.sample(batch_size)
        states = np.array([e[0] for e in experiences])
        rewards = np.array([e[2] for e in experiences])

        # Compute the advantages
        advantages = rewards - np.mean(rewards)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        with tf.GradientTape() as tape:
            # Forward pass of the model
            logits = self.policy_gradient_model(states.squeeze())
            logits = tf.reshape(logits, (-1, self.action_size))
            log_probs = tf.nn.log_softmax(logits)

            # Sample actions from the policy
            sampled_actions = tf.random.categorical(log_probs, 1)

            # Compute the negative log likelihood of the actions taken
            negative_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf.one_hot(sampled_actions,
                                                                                                self.action_size))
            # Compute the loss as the mean of the negative log likelihood multiplied by the rewards
            loss = tf.reduce_mean(negative_log_likelihood * advantages)
        # Compute the gradients of the loss with respect to the model's trainable weights
        grads = tape.gradient(loss, self.policy_gradient_model.trainable_weights)
        # Apply the gradients to the model's optimizer
        self.policy_gradient_model.optimizer.apply_gradients(zip(grads, self.policy_gradient_model.trainable_weights))

        # Keep track of the loss in TensorBoard
        with tf.summary.create_file_writer(self.policy_gradient_log).as_default():
            tf.summary.scalar('loss', loss, step=self.policy_gradient_model.optimizer.iterations)

    def choose_task(self, state, job):
        task = self.master_model.predict(state)

        if job == 'train' and np.random.rand() <= self.dqn_epsilon:
            self.dqn_epsilon *= self.dqn_epsilon_decay
            self.dqn_epsilon = max(self.dqn_epsilon, self.dqn_epsilon_min)
            return random.randrange(len(self.tasks.keys()))

        return np.argmax(task)

    def act(self, state, task, job='test'):
        """Method for getting the next action for the agent to take"""
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

    def save_models_and_memories(self, folder_name='model2'):
        if not os.path.exists(folder_name):
            # Create the folder if it does not exist
            os.mkdir(folder_name)
        models = {'dqn': self.dqn_model,
                  'ddqn': self.ddqn_model,
                  'ddqn_target': self.ddqn_target_model,
                  'actor_critic': self.actor_critic_model,
                  'policy_gradient': self.policy_gradient_model}
        memories = {'dqn': self.dqn_memory,
                    'ddqn': self.ddqn_memory,
                    'actor_critic': self.actor_critic_memory,
                    'policy_gradient': self.policy_gradient_memory}
        self.master_model.save(os.path.join(folder_name, f"master.h5"))
        with open(os.path.join(folder_name, 'master_memories.pickle'), 'wb') as handle:
            pickle.dump(self.master_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for model_name, model in models.items():
            model.save(os.path.join(folder_name, f"{model_name}.h5"))
        with open(os.path.join(folder_name, 'memories.pickle'), 'wb') as handle0:
            pickle.dump(memories, handle0, protocol=pickle.HIGHEST_PROTOCOL)

    def load_performance_and_counters(self, folder_name='model2'):
        if os.path.exists(folder_name):
            with open(os.path.join(folder_name, 'master_memories.pickle'), 'rb') as handle0:
                self.master_memory = pickle.load(handle0)
        else:
            return

    def load_models_and_memories(self, name='trial1', folder_name='model2'):
        if os.path.exists(folder_name):
            for task in ['dqn', 'ddqn', 'actor_critic', 'policy_gradient']:
                if task == 'dqn' and os.path.exists(f'{folder_name}/{name}_dqn.h5'):
                    self.dqn_model.load_weights(os.path.join(folder_name, f'{name}_dqn.h5'))
                    with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                        self.dqn_memory = pickle.load(f)
                elif task == 'ddqn' and os.path.exists(f'{folder_name}/{name}_ddqn.h5'):
                    self.ddqn_model.load_weights(os.path.join(folder_name, f'{name}_ddqn.h5'))
                    self.ddqn_target_model.load_weights(os.path.join(folder_name, f'ddqn_target.h5'))
                    with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                        self.ddqn_memory = pickle.load(f)
                elif task == 'actor_critic' and os.path.exists(f'{folder_name}/{name}_actor_critic.h5'):
                    self.actor_critic_model.load_weights(os.path.join(folder_name, f'{name}_actor_critic.h5'))
                    with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                        self.actor_critic_memory = pickle.load(f)
                elif task == 'policy_gradient' and os.path.exists(f'{folder_name}/{name}_policy_gradient.h5'):
                    self.policy_gradient_model.load_weights(os.path.join(folder_name, f'{name}_policy_gradient.h5'))
                    with open(os.path.join(folder_name, f'{name}_{task}_memory.pickle'), 'rb') as f:
                        self.policy_gradient_memory = pickle.load(f)
        else:
            return

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
        elif task == 'ddqn':
            self.add_ddqn_transition(state, action, reward, next_state, done)
        elif task == 'actor_critic':
            self.add_actor_critic_transition(state, action, reward, next_state, done)
        elif task == 'policy_gradient':
            self.add_policy_gradient_transition(state, action, reward, next_state, done)

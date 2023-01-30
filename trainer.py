# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 01:45:26 2023

@author: alial
"""

from collections import deque
import matplotlib.pyplot as plt
from trader import Trader, Market
from agents import MultiTask
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import gc


def train(task):
    # Initialize the Trader object and connect to the IB gateway
    levels = {0: 0.005, 1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.1, 7: 0.15, 8: 0.2, 9: 0.25}
    # Update learning of models
    
    trader = Trader()

    # Initialize the DQN agent
    action_size = 5

    previous_row = None

    rewards = [0]
    steps = [0]
    # Initialize the rolling window to store the last n rewards
    window_size = 100
    rolling_window = deque(maxlen=window_size)
    # Initialize the list to store the rolling average of the rewards, profits, iterations and batch size levels
    rolling_average = []
    current_batch_size_level = 0
    current_iteration = 0
    batch_size = 10

    # while not trader.profit >= 1000000 * 0.3 or not trader.num_trades >= 1000:
    # start first iteration
    current_iteration += 1
    # initialize market and get the dataframe
    market = Market(trader, history_length=1)
    market.update_data()
    df = market.get_df()
    # Get the current and next states
    state = market.get_state(0)
    state_size = state.shape[1]

    agent = MultiTask(task=task, state=state, action_size=action_size, state_size=state_size, job='train')
    replay_functions = {
        "dqn": agent.replay_dqn,
        "ddqn": agent.replay_ddqn,
        "actor_critic": agent.replay_actor_critic,
        "policy_gradient": agent.replay_policy_gradient
    }
    # Load saved trainings and memories from previous sessions
    if eval(f'agent.{task}_memory._size()') == 0:
        agent.load(name='trial1', task=task)
    for level, percentage in levels.items():
        if int(len(df) * percentage) < eval(f'agent.{task}_memory._size()'):
            batch_size = int(len(df) * percentage)
    while not trader.profit >= 1000000 * 0.3 or not trader.num_trades >= 1000:
        for i, row in df.iterrows():
            if previous_row is None:
                previous_row = row
            done = i + 2 >= len(df)
            if done:
                break
            if i == int(len(df) * levels[current_batch_size_level]) and batch_size <= len(df) * levels[
                current_batch_size_level]:
                batch_size = int(len(df) * levels[current_batch_size_level])
                print(
                    f'Level {list(levels.keys())[current_batch_size_level]} is done. Batch size now is {batch_size} '
                    f'({levels[current_batch_size_level] * 100}% of the data)')

                if current_batch_size_level <= next(reversed(levels.items()))[0] - 1:
                    current_batch_size_level += 1
            # Get nextstate value
            next_state = market.get_state(i + 1)

            # Predict the action using the model
            action = agent.act(state=state, task=task, job='train')
            # Execute the trade and get the reward
            reward = trader.trade(action, row, previous_row, i)
            # Append the total reward and number of steps for this episode to the lists
            rewards.append(reward)
            steps.append(i)
            agent.add_to_memory(task, state, action, reward, next_state, done)
            state = next_state
            if i>2000: break
            rolling_window.append(trader.realized_profit_loss)
            # Calculate the rolling average of the rewards and append it to the list
            rolling_average.append(np.mean(rolling_window))
            # Set the current state to the next state
            plt.plot(rolling_average)
            previous_row = row

            replay_functions[task](batch_size)

            if i % 50 == 0:
                # Save the update agents
                agent.save('trial1', task)
                # Garbage data disposal
                gc.collect()
                # Print progress
                print(
                    f"***************** Episode {current_iteration} of {task} final account was {trader.total_value}"
                    f" which is a total profit/loss of {trader.total_value - trader.capital}")
    return trader.total_value - trader.capital


if __name__ == '__main__':
    # results = train(task="dqn")
    with Pool(4) as p:
        results = [p.map(train, ['dqn', 'ddqn', 'actor_critic', 'policy_gradient'])]
        print(results)

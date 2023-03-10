# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 01:45:26 2023

@author: alial
"""

from collections import deque
import matplotlib.pyplot as plt
from trader import Trader, Market
<<<<<<< HEAD
from agentsEpsGreed import MultiTask
=======
from agents import MultiTask
>>>>>>> 97b72eb88e95ec18daaf333344a9334aa757c634
import numpy as np
from multiprocessing import Pool
import gc


def test(task):
    # Initialize the Trader object and connect to the IB gateway
    # print('*********')
    trader = Trader()

    # Initialize the DQN agent
    action_size = 5
    batch_size = 100
    previous_row = None

    rewards = [0]
    steps = [0]
    # Initialize the rolling window to store the last n rewards
    window_size = 100
    rolling_window = deque(maxlen=window_size)
    # Initialize the list to store the rolling average of the rewards
    rolling_average = []
    # Update market data and get the DataFrame
<<<<<<< HEAD
    market = Market(trader)
=======
    market = Market(trader, history_length=1)
>>>>>>> 97b72eb88e95ec18daaf333344a9334aa757c634
    market.update_data()
    df = market.get_df()
    state = market.get_state(0)
    state_size = state.shape[1]
<<<<<<< HEAD
    agent = MultiTask(task=task, state_size=state_size, action_size=action_size, job='test')
=======
    agent = MultiTask(task=task, state=state, state_size=state_size, action_size=action_size, job='test')
>>>>>>> 97b72eb88e95ec18daaf333344a9334aa757c634
    if eval(f'agent.{task}_memory._size()') == 0:
        agent.load(name='trial1', task=task)

    for i, row in df.iterrows():
        if previous_row is None:
            previous_row = row
        done = i + 2 >= len(df)
        if done:
            break

        # Get nextstate value
        next_state = market.get_state(i + 1)

        # Predict the action using the model
        action = agent.act(state=state, task=task, job='test')
        print(f'iterate {i} of {task} yielded {action}')
        # Execute the trade and get the reward
        reward = trader.trade(action, row, previous_row, i)
        rewards.append(reward)
        steps.append(i)

        rolling_window.append(trader.realized_profit_loss)

        # Calculate the rolling average of the rewards and append it to the list
        rolling_average.append(np.mean(rolling_window))

        # Set the current state to the next state
        state = next_state

        # Garbage data disposal
        gc.collect()

    # Create and plot a graph to show agent profits
    plt.plot(rolling_average)
    plt.show()
    print(f'Agent performance in test session yielded ${round((trader.total_value - trader.capital), 2)} return')
    return trader.total_value - trader.capital


if __name__ == '__main__':
<<<<<<< HEAD
    results = test(task="actor_critic")
=======
    results = test(task="dqn")
>>>>>>> 97b72eb88e95ec18daaf333344a9334aa757c634
    # with Pool(4) as p:
    #     results = [p.map(test, ['dqn', 'ddqn', 'actor_critic', 'policy_gradient'])]
    #     print(results)

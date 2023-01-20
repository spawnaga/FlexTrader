# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 01:45:26 2023

@author: alial
"""

from collections import deque
import matplotlib.pyplot as plt
from trader import Trader, Market
from ib_insync import ContFuture
from agents import MultiTask
import numpy as np
from multiprocessing import Pool


def test(task):
    # Initialize the Trader object and connect to the IB gateway
    # print('*********')
    trader = Trader()
    contract = ContFuture('NQ', 'CME')
    trader.ib.qualifyContracts(contract)

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
    pnl = [0]


    # while not trader.profit >= 1000000*0.3 or not trader.num_trades >= 1000:
    market = Market(trader, contract, history_length=1)
    market.update_data()
    df = market.get_df()
    state = market.get_state(0).reshape(-1, 7)
    state_size = state.shape[1]
    agent = MultiTask(task=task, state=state, num_outputs_1=action_size, num_outputs_2=action_size,
                      state_size=state_size, action_size=action_size)
    if not eval(f'agent.{task}_memory'):
        agent.load(name= 'trial1', task= task)
        if task == "dqn":
            agent.dqn_epsilon=-1
        elif task =='ddqn':
            agent.ddqn_epsilon=-1
    # if agent.dqn_memory ==deque([]):
    #     agent.load("trial1",task)
    for i, row in df.iterrows():
        if previous_row is None:
            previous_row = row
        done = i + 2 >= len(df)
        if done:
            break
        # if i>35: break

        # Get the current and next states
        next_state = market.get_state(i + 1).reshape(-1, 7)
        # Predict the action using the model
        action = agent.act(state=state.reshape(-1, 7), task=task, job='test')
        # Execute the trade and get the reward
        reward = trader.trade(contract, action, market, i, row, previous_row)
        # Append the total reward and number of steps for this episode to the lists
        rewards.append(reward)
        steps.append(i)
        agent.add_to_memory(task, state, action, reward, next_state, done)
        # if i>35: break

        rolling_window.append(trader.realized_profit_loss)

        # Calculate the rolling average of the rewards and append it to the list
        rolling_average.append(np.mean(rolling_window))

        # Set the current state to the next state
        state = next_state


        plt.plot(rolling_average)


        print(f"***************** iteration {i} of {task} final account was {trader.total_value} which is a total profit/loss of {trader.total_value - trader.capital}")
    plt.show()
    return pnl


if __name__ == '__main__':
    # for i in range(10):
    with Pool(4) as p:
        results = [p.map(test, ['dqn', 'ddqn', 'actor_critic', 'policy_gradient'])]
        print(results)



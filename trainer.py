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

def train(task):
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
    j=0
    # Update market data and get the DataFrame
    pnl=[0]
    while not trader.profit >= 1000000*0.3 or not trader.num_trades >= 1000:
        j+=1
        # while not trader.profit >= 1000000*0.3 or not trader.num_trades >= 1000:
        market = Market(trader, contract, history_length=1)
        market.update_data()
        df = market.get_df()
        state = market.get_state(0)
        state_size = state.shape[1]
        agent = MultiTask(task=task, state=state, num_outputs_1=action_size, num_outputs_2 = action_size, state_size=state_size, action_size=action_size)
        # Load saved trainings and memories from previous sessions
        if not eval(f'agent.{task}_memory'):
            agent.load(name='trial1', task=task)
        for i, row in df.iterrows():
            if previous_row is None:
                previous_row = row
            done = i + 2 >= len(df)
            if done:
                break
            # if i>35: break
    
            # Get the current and next states
            next_state = market.get_state(i+1)
            # Predict the action using the model
            action = agent.act(state=state, task= task, job='train')
            print(f'iterate {i} of {task} yielded {action}')
            # Execute the trade and get the reward
            reward = trader.trade(contract, action, market, i,row, previous_row)
            # Append the total reward and number of steps for this episode to the lists
            rewards.append(reward)
            steps.append(i)
            agent.add_to_memory(task,state, action, reward, next_state, done)
            # if i>35: break
    
            rolling_window.append(trader.realized_profit_loss)
    
            # Calculate the rolling average of the rewards and append it to the list
            rolling_average.append(np.mean(rolling_window))
    
            # Set the current state to the next state
            state = next_state
    
        # if trader.total_value > max(pnl):
        #     pnl.append(trader.total_value)
        #     # Calculate the mean and standard deviation of the rewards and steps
        #     mean_reward = np.mean(rewards)
        #     std_reward = np.std(rewards)
        #     mean_steps = np.mean(steps)
        #     std_steps = np.std(steps)

            # Print the results
            # print(f"Mean reward per episode: {mean_reward:.2f} +/- {std_reward:.2f}")
            # print(f"Mean steps per episode: {mean_steps:.2f} +/- {std_steps:.2f}")
            plt.plot(rolling_average)
            # plt.show()

            replay_functions = {
                                "dqn": agent.replay_dqn,
                                "ddqn": agent.replay_ddqn,
                                "actor_critic": agent.replay_actor_critic,
                                "policy_gradient": agent.replay_policy_gradient
                            }
            replay_functions[task](batch_size)
            agent.save('trial1', task)
            print(f"***************** Episode {j} of {task} final account was {trader.total_value} which is a total profit/loss of {trader.total_value - trader.capital}")
    return pnl


if __name__ == '__main__':
    # for i in range(10):
    with Pool(20) as p:
        results = [p.map(train, ['dqn', 'ddqn', 'actor_critic', 'policy_gradient'])]
        print(results)



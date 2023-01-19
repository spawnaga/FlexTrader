from collections import deque
import matplotlib.pyplot as plt
from trader import Trader, Market
from ib_insync import ContFuture
from agents import MultiTask
import numpy as np
        
def train():
    # Initialize the Trader object and connect to the IB gateway
    trader = Trader()
    contract = ContFuture('NQ', 'CME')
    trader.ib.qualifyContracts(contract)
    
    # Initialize the DQN agent
    
    action_size = 5
    
    task = ['dqn', 'ddqn', 'actor_critic','policy_gradient']
    
    batch_size = 100
    
    # Update market data and get the DataFrame

    while not trader.profit >= 1000000*0.3 or not trader.num_trades >= 1000:
        
        # Initialize the Market object
        market = Market(trader, contract, history_length=1)
        market.update_data()
        df = market.get_df()
        state = market.get_state(0).reshape(-1,7)
        state_size = state.shape[1]
        agent = MultiTask(state=state, num_outputs_1=action_size, num_outputs_2 = action_size, state_size=state_size, action_size=action_size)

        previous_row = None

        rewards = [0]
        steps = [0]
        # Initialize the rolling window to store the last n rewards
        window_size = 100
        rolling_window = deque(maxlen=window_size)
        # Initialize the list to store the rolling average of the rewards
        rolling_average = []
        if agent.dqn_memory ==deque([]):
            agent.load("trial1")

        for i, row in df.iterrows():
            
            if previous_row is None:
                previous_row = row
            done = i + 2 >= len(df) 
            if done:
                break
            # if i>35: break
            
            # Get the current and next states
            next_state = market.get_state(i+1).reshape(-1,7)
            
            for _ in task:
                # Predict the action using the model
                action = agent.act(state=state.reshape(-1,7), task= _)
                
                # Execute the trade and get the reward
                reward = trader.trade(contract, action, market, i,row, previous_row)
                # Append the total reward and number of steps for this episode to the lists
                rewards.append(reward)
                steps.append(i)
                agent.add_to_memory(_,state, action, reward, next_state, done)
                # if i>35: break

            rolling_window.append(round(trader.total_value,2))
            
            # Calculate the rolling average of the rewards and append it to the list
            rolling_average.append(np.mean(rolling_window))
            
            # Set the current state to the next state
            state = next_state
            
            if i%50==0 :
                # Calculate the mean and standard deviation of the rewards and steps
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                mean_steps = np.mean(steps)
                std_steps = np.std(steps)
                
                # Print the results
                print(f"Mean reward per episode: {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Mean steps per episode: {mean_steps:.2f} +/- {std_steps:.2f}")
                plt.plot(rolling_average)
                plt.show()
            # if i%100==0:
                agent.save('trial1')
                agent.replay_dqn(batch_size)
                agent.replay_ddqn(batch_size)
                agent.replay_actor_critic(batch_size)
                agent.replay_policy_gradient(batch_size)
            previous_row =row
        print(trader.profit, trader.num_trades)
    
    # Print performance metrics
    print(f'Total profit: {trader.profit}')
    print(f'Number of trades: {trader.num_trades}')
    print(f'Number of wins: {trader.num_wins}')
    print(f'Number of losses: {trader.num_losses}')
    # agent.save()
    trader.disconnect()

if __name__ == '__main__':

    train()
#     # test()

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from collections import deque
from ib_insync import *
from indicators import indicators_dataframe
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Start the asyncio event loop
util.startLoop()

class Trader:
    """Class for interacting with Interactive Brokers Gateway"""
    def __init__(self, history_length=10):
        # Connect to IB gateway
        self.ib = IB()
        self.ib.connect('192.168.0.78', 7496, clientId=np.random.randint(0,1000))
        
        # Initialize attributes
        self.profit = 0
        self.num_trades = 0
        self.num_wins = 0
        self.num_losses = 0
        self.max_profit = 0
        self.max_loss = 0
        self.cash = 1000000
        self.num_contracts = 0
        self.total_value =self.cash+(self.num_contracts*20)
        self.close_prices=[]
        self.trades=[]
        self.priceAtStart=0
        self.priceAtClose=0

        # Initialize the list to store total_value at each time step
        self.total_value_history = []
   
        
    def trade(self, contract, action, market, i,row, previous_row):
        """Execute a trade based on the current market state and the output of the model"""
        
        
        # Get close price from Market object
        close_price = row['close']
        previous_close_price = previous_row['close']
        
        
        # Calculate number of contracts to trade using fixed-fractional method
        if self.num_contracts ==0 and action ==0:
            self.num_contracts = int(self.cash / close_price/20)
            self.cash = self.cash - abs(self.num_contracts*close_price*20)
            self.priceAtStart=close_price
            self.num_trades += 1
            Action = "buy"
        elif self.num_contracts ==0 and action ==1:
            self.num_contracts = -1*int(self.cash / close_price/20)
            # self.total_value =self.cash+abs(self.num_contracts*20)
            self.cash = abs(-1*self.cash - self.num_contracts*close_price*20)
            self.priceAtStart=close_price
            self.num_trades += 1
            Action = "sell"
        elif self.num_contracts >0 and action ==3:      
            self.cash = abs(self.total_value)
            
            self.priceAtClose=close_price
            Action = "sell to close Longs"
            print(f"*******Trade {self.num_trades} got {round((self.priceAtClose-self.priceAtStart)*20* self.num_contracts,2)} return*******")
            self.num_contracts = 0
            if self.priceAtClose-self.priceAtStart > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
        elif self.num_contracts <0 and action ==4:
            self.cash = abs(self.total_value)
            
            self.priceAtClose=close_price
            Action = "buy to close Shorts"
            print(f"*******Trade {self.num_trades} got {round((self.priceAtStart-self.priceAtClose)*20*self.num_contracts,2)} return*******")
            self.num_contracts = 0
            if self.priceAtStart-self.priceAtClose > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
        else:       
            action = 2
            Action = "hold"
        
        self.close_prices.append(close_price)
        self.trades.append(action)
        # Calculate profit or loss of trade
        profit_loss = self.num_contracts * (close_price - previous_close_price) *20
        self.total_value += profit_loss
        # Update performance metrics
        self.profit += profit_loss
         #if action == 0 or action == 1 else 0
        
        self.max_profit = max(self.max_profit, profit_loss)
        self.max_loss = min(self.max_loss, profit_loss)
        print(f'Trade: {self.num_trades}, last close price: {close_price}, previous close price: {previous_close_price}, Action: {Action}, prices differences: { close_price - previous_close_price}; Profit/Loss: {round(profit_loss,2)}, Total PNL: {round(self.profit,2)}, Account NQ contracts holding = {self.num_contracts}, Account cash balance = {round(self.cash,2)}, Account total value = {round(self.total_value,2)}')

        return profit_loss

    
    def disconnect(self):
        """Method for disconnect"""
        self.ib.disconnect()



class Market:
    """Class for handling market data"""
    def __init__(self, trader, contract, history_length=1):
        # Store the history length as an instance variable
        self.history_length = history_length
        self.trader = trader
        # Get the contract and data
        self.contract = contract
        self.scaler = MinMaxScaler(feature_range=(0, 4))
        trader.ib.qualifyContracts(self.contract)
        
    def update_data(self):
        # Download historical data using reqHistoricalData
        self.bars = self.trader.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr=f'{self.history_length} M',
            barSizeSetting='5 mins', whatToShow='TRADES',
            useRTH=False
        )
        
        # Create a DataFrame from the downloaded data
        df = util.df(self.bars)
        # df = self.get_analysis(df)
        df.reset_index(inplace = True,drop=True)
        # df = df.drop(df.iloc[:,10:],axis=1)
        df['contract']=0
        df=df.drop(['date','average'],axis=1)
        self.df = df
        self.data = self.scaler.fit_transform(df)
        
    def get_state(self, i=0):
        """Method for getting current state of market"""
        state = self.data[i]
        contracts_holding = self.trader.num_contracts
        state[-1]=contracts_holding
        np.append(state, i + 2 >= len(self.data))
        return np.expand_dims(state, 0)
    
    def get_analysis(self, df):
        # Analyze using Technical Analysis on talib libraries
        df = indicators_dataframe(df)
        df = df[df.columns[4:]]
        df = df.drop('volume',axis=1)
        df = df.dropna()
        return df
    
    def get_df(self):
        """Method for returning the DataFrame of market data"""
        return self.df


class Memory:
    def __init__(self, max_size=32):
        self.max_size = max_size
        self.memory = []

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.max_size:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def _size(self):
        return len(self.memory)
    
    def clear(self):
        self.memory = []
    

class MultiTask:
    def __init__(self, state, state_size, action_size, num_outputs_1=5, num_outputs_2=5):
      # Initialize attributes
      self.state = state
      self.state_size = state_size
      self.action_size = action_size
      self.learning_rate = 0.001
    
      # Initialize DQN model
      self.dqn_model = self._build_dqn_model(num_outputs_1, num_outputs_2)
      self.dqn_memory = Memory(max_size=10000)
      self.dqn_gamma = 0.95
      self.dqn_epsilon = 1.0
      self.dqn_epsilon_min = 0.01
      self.dqn_epsilon_decay = 0.995
      self.dqn_learning_rate = 0.001
    
      # Initialize actor-critic model
      self.actor_critic_model = self._build_actor_critic_model(num_outputs_1, num_outputs_2)
      self.actor_critic_memory = Memory(max_size=10000)
      self.actor_critic_gamma = 0.95
      self.actor_critic_alpha = 0.001
      self.actor_critic_alpha_decay = 0.995
      self.actor_critic_alpha_min = 0.01 
    
      # Initialize policy gradient model
      self.policy_gradient_model = self._build_policy_gradient_model(num_outputs_1, num_outputs_2)
      self.policy_gradient_memory = Memory(max_size=10000)
      self.policy_gradient_gamma = 0.95
      self.policy_gradient_learning_rate = 0.001

    def add_dqn_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the DQN task
        transition = (state, action, reward, next_state, done)
        self.dqn_memory.add(transition)
    
    def add_actor_critic_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the actor-critic task
        transition = (state, action, reward, next_state, done)
        self.actor_critic_memory.add(transition)
    
    def add_policy_gradient_transition(self, state, action, reward, next_state, done):
        # Add a transition to the memory for the policy gradient task
        transition = (state, action, reward, next_state, done)
        self.policy_gradient_memory.add(transition)

    def _build_dqn_model(self, num_outputs_1, num_outputs_2):
        #setting up the model of tensorflow
        input_shape = (self.state_size)
   
        # Define the input layer
        input_layer = Input((None,input_shape))
        x=input_layer
        for _ in range(5): # five layers
               # x = Dropout(0.2)(x) # Dropout to avoid overfitting
               x = CuDNNLSTM(self.state.shape[1]*2, return_sequences = True)(x) # using LSTM with return sequences to adopt to time sequences
               # x = Dense(5, activation = 'relu')(x)
        x = GlobalAveragePooling1D()(x) # Global averaging to one layer shape to feed to a dense categorigal classification
        output = Dense(units=self.action_size, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse',  optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999))
        print((model.summary()))
  
        return model

    def _build_actor_critic_model(self, num_outputs_1, num_outputs_2):
        #setting up the model of tensorflow
        input_shape = (self.state_size)

        # Define the input layer
        input_layer = Input((None,input_shape))
        x=input_layer
        for _ in range(5): # five layers
               # x = Dropout(0.2)(x) # Dropout to avoid overfitting
               x = CuDNNLSTM(self.state.shape[1]*2, return_sequences = True)(x) # using LSTM with return sequences to adopt to time sequences
               # x = Dense(5, activation = 'relu')(x)
        x = GlobalAveragePooling1D()(x) # Global averaging to one layer shape to feed to a dense categorigal classification
        output = Dense(units=self.action_size, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse',  optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999))
        print((model.summary()))
    
        return model

    def _build_policy_gradient_model(self, num_outputs_1, num_outputs_2):
        #setting up the model of tensorflow
        input_shape = (self.state_size)

        # Define the input layer
        input_layer = Input((None,input_shape))
        x=input_layer
        for _ in range(5): # five layers
               # x = Dropout(0.2)(x) # Dropout to avoid overfitting
               x = CuDNNLSTM(self.state.shape[1]*2, return_sequences = True)(x) # using LSTM with return sequences to adopt to time sequences
               # x = Dense(5, activation = 'relu')(x)
        x = GlobalAveragePooling1D()(x) # Global averaging to one layer shape to feed to a dense categorigal classification
        output = Dense(units=self.action_size, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse',  optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999))
        print((model.summary()))
    
        return model
    
    def replay_dqn(self, batch_size):
        
        if self.dqn_memory._size() < batch_size:
            return
    
        # Sample a batch of transitions from the memory
        batch = self.dqn_memory.sample( batch_size)
    
        # Initialize lists for the states, actions, rewards, next states, and dones
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
    
        # Loop through the transitions in the batch and add them to the lists
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
    
        # Use the DQN model to predict the target Q-values for the next states
        target_q_values = np.expand_dims(self.dqn_model.predict(np.array(next_states)),0)
    
        # Initialize a list for the updated Q-values
        updated_q_values = []
    
        # Loop through the transitions in the batch and update the Q-values
        for i, done in enumerate(dones):
            # If the episode is done, set the Q-value for the action to the reward
            if done:
                updated_q_values.append(rewards[i])
            # If the episode is not done, set the Q-value for the action to the reward plus the discounted value of the next state
            else:
                max_q_value = np.amax(target_q_values[0][i])
                updated_q_value = rewards[i] + self.dqn_gamma * max_q_value
                updated_q_values.append(updated_q_value)
                
        # Convert the states, actions, and updated Q-values to arrays
        states = np.array(states)
        actions = np.array(actions)
        
        actions = to_categorical(actions, num_classes=self.action_size)
        actions = np.expand_dims(actions, axis=1)
        updated_q_values = np.array(updated_q_values)
        
        # Fit the DQN model on the states, actions, and updated Q-values

        self.dqn_model.fit(x=states, y=actions, sample_weight=updated_q_values, verbose=0)
        
        # Decrease the exploration rate
        self.dqn_epsilon *= self.dqn_epsilon_decay
        
        # If the exploration rate is less than the minimum exploration rate, set it to the minimum exploration rate
        self.dqn_epsilon = max(self.dqn_epsilon_min, self.dqn_epsilon)

            
    def replay_actor_critic(self, batch_size):
        # Check if there are enough transitions in the memory to create a batch
        if self.actor_critic_memory._size() < batch_size:
            return
    
        # Sample a batch of transitions from the memory
        batch = self.actor_critic_memory.sample(batch_size)
    
        # Initialize lists for the states, actions, rewards, and dones
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
    
        # Loop through the transitions in the batch and add them to the lists
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
    
        # Use the actor-critic model to predict the value functions for the states
        values = self.actor_critic_model.predict(np.array(states))[1]
    
        # Initialize lists for the updated value functions and advantages
        updated_values = []
        advantages = []
    
        # Loop through the transitions in the batch and update the value functions and advantages
        for i, done in enumerate(dones):
            # If the episode is done, set the value function to the reward
            if done:
                updated_values.append(rewards[i])
            # If the episode is not done, set the value function to the reward plus the discounted value of the next state
            else:
                print(len(rewards))
                print(values.shape)
                updated_values.append(rewards[i] + self.actor_critic_gamma * values[i])
            # Calculate the advantage for the action
            advantages.append(updated_values[i] - values[i])
    
        # Convert the states, actions, and advantages to arrays
        states = np.array(states)
        actions = np.array(actions)
        actions = to_categorical(actions, num_classes=self.action_size)
        actions = np.expand_dims(actions, axis=1)
        advantages = np.array(advantages)
        
        # Standardize the advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        updatedValues = np.array(updated_values)
        self.actor_critic_model.fit(x=[states], y=[actions, updatedValues, advantages], verbose=0)
        
        # Clear the actor-critic memory
        self.actor_critic_memory.clear()

    def replay_policy_gradient(self, batch_size):
        # Check if there are enough transitions in the memory to create a batch
        if self.actor_critic_memory._size() < batch_size:
            return
    
        # Sample a batch of transitions from the memory
        batch = self.actor_critic_memory.sample(batch_size)
    
        # Initialize lists for the states, actions, and rewards
        states = []
        actions = []
        rewards = []
    
        # Loop through the transitions in the batch and add them to the lists
        for state, action, reward in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
    
        # Convert the states and actions to arrays
        states = np.array(states)
        actions = np.array(actions)
    
        # Use the policy gradient model to predict the action probabilities for the states
        action_probs = self.policy_gradient_model.predict(states)
    
        # Initialize a list for the log probabilities of the actions
        log_probs = []
    
        # Loop through the transitions in the batch and calculate the log probability of the action
        for i, action in enumerate(actions):
            log_prob = np.log(action_probs[i][action])
            log_probs.append(log_prob)
    
        # Calculate the total reward for each transition
        total_rewards = []
        for i in range(len(rewards)):
            # Calculate the total reward for the transition
            total_reward = sum([rewards[i] * self.policy_gradient_gamma**j for j in range(len(rewards[i:]))])
            total_rewards.append(total_reward)
    
        # Standardize the total rewards
        total_rewards = (total_rewards - np.mean(total_rewards)) / (np.std(total_rewards) + 1e-10)
    
        # Convert the log probabilities and total rewards to arrays
        log_probs = np.array(log_probs)
        total_rewards = np.array(total_rewards)


    def predict_dqn(self, state):
        # If the exploration rate is greater than a random value, return a random action
        if np.random.rand() <= self.dqn_epsilon:
            return random.randrange(self.action_size)
        
        # Use the DQN model to predict the action with the highest expected reward for the given state
        action = self.dqn_model.predict(state)
        return np.argmax(action)  # Return the action with the highest expected reward

    
    def predict_actor_critic(self, state):
        # Use the actor-critic model to predict the action probabilities and value function for the given state
        action = np.argmax(self.actor_critic_model.predict(state))
        return action

    
    def predict_policy_gradient(self, state):
        # Use the policy_gradient_model to predict the action probabilities for the given state
        action = np.argmax(self.policy_gradient_model.predict(state))
        return action
    

    def select_task(self, state):
        # Use the DQN model to predict the best action for the state
        dqn_best_action = self.predict_dqn(state)
    
        # Use the actor-critic model to predict the best action for the state
        actor_critic_best_action= self.predict_actor_critic(state)
    
        # Use the policy gradient model to predict the best action for the state
        policy_gradient_best_action = np.argmax(self.predict_policy_gradient(state))
    
        # Initialize a dictionary for the scores of the tasks
        scores = {}
    
        # Calculate the score for the DQN task
        dqn_score = self.dqn_epsilon / (self.dqn_epsilon + 0.1)
        scores['dqn'] = dqn_score
    
        # Calculate the score for the actor-critic task
        actor_critic_score = self.actor_critic_alpha / (self.actor_critic_alpha + 0.1)
        scores['actor_critic'] = actor_critic_score
    
        # Calculate the score for the policy gradient task
        policy_gradient_score = 1 - dqn_score - actor_critic_score
        scores['policy_gradient'] = policy_gradient_score
    
        # Select the task with the highest score
        selected_task = max(scores, key=scores.get)
    
        # Return the best action for the selected task
        if selected_task == 'dqn':
            return dqn_best_action
        elif selected_task == 'actor_critic':
            return actor_critic_best_action
        elif selected_task == 'policy_gradient':
            return policy_gradient_best_action

    def add_to_memory(self, task, state, action, reward, next_state, done):
        # Add the transition to the memory for the specified task
        if task == 'dqn':
            self.add_dqn_transition(state, action, reward, next_state, done)
        elif task == 'actor_critic':
            self.add_actor_critic_transition(state, action, reward, next_state, done)
        elif task == 'policy_gradient':
            self.add_policy_gradient_transition(state, action, reward, next_state, done)

    def act(self, state):
        # Select the task and get the best action for the state
        if state.shape != (None,1,7):
            state=state.reshape(-1,1,7)
        action = self.select_task(state)
    
        # Decrease the exploration rate for the DQN task
        self.dqn_epsilon *= self.dqn_epsilon_decay
        self.dqn_epsilon = max(self.dqn_epsilon_min, self.dqn_epsilon)
    
        # Decrease the exploration rate for the actor-critic task
        self.actor_critic_alpha *= self.actor_critic_alpha_decay
        self.actor_critic_alpha = max(self.actor_critic_alpha_min, self.actor_critic_alpha)
    
        return action


        
        
def train():
    # Initialize the Trader object and connect to the IB gateway
    trader = Trader()
    contract = ContFuture('NQ', 'CME')
    trader.ib.qualifyContracts(contract)
    
    # Initialize the DQN agent
    
    action_size = 5
    
    task = ['dqn','actor_critic','policy_gradient']
    
    batch_size = 32
    
    # Update market data and get the DataFrame
  

    while trader.profit<1000000*0.3:
        
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
        # if agent.memory ==deque([]):
        #     agent.load("dqna")
        # Loop over the historical data
        for i, row in df.iterrows():
            if previous_row is None:
                previous_row = row
            # if i>5:break
            done = i + 2 >= len(df) 
            if done:
                break
            # Get the current and next states
            next_state = market.get_state(i+1).reshape(-1,7)
            
            # Predict the action using the model
            action = agent.act(state=state.reshape(-1,1,7))
            
            # Execute the trade and get the reward
            reward = trader.trade(contract, action, market, i,row, previous_row)
            # Append the total reward and number of steps for this episode to the lists
            rewards.append(reward)
            steps.append(i)
            
            # Store the experience in the memory and train the agent
            for _ in task:
                agent.add_to_memory(_,state, action, reward, next_state, done)
            
            # Update the models using the memories
            agent.replay_dqn(batch_size)
            agent.replay_actor_critic(batch_size)
            agent.replay_policy_gradient(batch_size)
            
            # Append the total reward for this episode to the rolling window
            rolling_window.append(reward)
            
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
            previous_row =row

            
        # Print performance metrics
        print(f'Total profit: {trader.profit}')
        print(f'Number of trades: {trader.num_trades}')
        print(f'Number of wins: {trader.num_wins}')
        print(f'Number of losses: {trader.num_losses}')
        print(f'Max profit: {trader.max_profit}')
        print(f'Max loss: {trader.max_loss}')

        # Calculate the mean and standard deviation of the rewards and steps
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_steps = np.mean(steps)
        std_steps = np.std(steps)
        
        # Print the results
        print(f"Mean reward per episode: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Mean steps per episode: {mean_steps:.2f} +/- {std_steps:.2f}")

        # Plot the rolling average of the rewards
        plt.plot(rolling_average)
        plt.show()
        # agent.epsilon =0.5
        agent.save("dgna")
        # Disconnect from the IB gateway
    trader.disconnect()


    
if __name__ == '__main__':

    train()
#     # test()

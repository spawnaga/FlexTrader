# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:46:39 2023

@author: alial
"""

# from indicators import indicators_dataframe
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from ib_insync import *
import time

util.startLoop()


class Trader:
    """Class for interacting with Interactive Brokers Gateway"""

    def __init__(self):

        # Initialize attributes
        self.profit = 0
        self.num_trades = 0
        self.num_wins = 0
        self.num_losses = 0
        self.max_profit = 0
        self.max_loss = 0
        self.capital = 1000000
        self.cash = 1000000
        self.num_contracts = 0
        self.total_value = self.cash + (self.num_contracts * 20)
        self.close_prices = []
        self.trades = []
        self.priceAtStart = 0
        self.priceAtClose = 0
        self.realized_profit_loss = 0
        self.unrealized_profit_loss = 0
        self.penalty = 1000
        self.expon = 1.05
        self.hold_penalty = -100
        self.last_action_time = 0
        self.punish_epsilon = -10000
        self.totInvalidPerc = 0
        self.counter = 0

    def _get_kelly_criterion(self, win_loss_ratio, avg_profit):
        """Calculate the optimal number of contracts to trade using Kelly Criterion"""
        return win_loss_ratio - (1 - win_loss_ratio) / (avg_profit / 20)

    def trade(self, action, row, previous_row, i, previous_action=2):
        """Execute a trade based on the current market state and the output of the model"""

        realizedPNL = 0
        # Get close price from Market object
        close_price = row['close']
        action_time = self.last_action_time
        # Calculate number of contracts to trade using fixed-fractional method
        if self.num_contracts <= 0 and action == 0:
            action_time = i
            if self.num_contracts < 0:
                self.priceAtClose = close_price
                state = "close short and start long position"
                realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
                self.num_contracts = 0
                self.cash = abs(self.total_value) + realizedPNL
                if realizedPNL > 0:
                    self.num_wins += 1
                else:
                    self.num_losses += 1
            state = "Long"
            self.num_contracts = int(self.cash / close_price / 20)
            self.cash = self.cash - abs(self.num_contracts * close_price * 20)
            self.priceAtStart = close_price
            self.num_trades += 1
        elif self.num_contracts >= 0 and action == 1:
            action_time = i
            if self.num_contracts > 0:
                self.priceAtClose = close_price
                state = "close long and start short position"
                realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
                self.num_contracts = 0
                self.cash = abs(self.total_value) + realizedPNL
                if realizedPNL > 0:
                    self.num_wins += 1
                else:
                    self.num_losses += 1
            state = "short"
            self.num_contracts = -1 * int(self.cash / close_price / 20)
            self.cash = abs(-1 * self.cash - self.num_contracts * close_price * 20)
            self.priceAtStart = close_price
            self.num_trades += 1
        elif self.num_contracts > 0 and action == 3:
            self.priceAtClose = close_price
            state = "close"
            realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
            self.num_contracts = 0
            self.cash = abs(self.total_value) + realizedPNL
            if realizedPNL > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
        elif self.num_contracts < 0 and action == 4:
            self.priceAtClose = close_price
            state = "close"
            realizedPNL = (self.priceAtStart - self.priceAtClose) * 20 * self.num_contracts
            self.num_contracts = 0
            self.cash = abs(self.total_value) + realizedPNL
            if realizedPNL > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
        elif self.num_contracts == 0 and action == 2:
            state = "hold"
        elif action == 2:
            state = "long" if self.num_contracts > 0 else "short"
        else:
            state = "invalid"
        if state == 'invalid' and i != 0:
            self.counter += 1
            self.totInvalidPerc = round(self.counter / i * 100, 2)

        rewards = round(
            self.calculate_rewards(state, row['close'], previous_row['close'], action, i, self.last_action_time,
                                   self.realized_profit_loss, previous_action), 2)
        if rewards < self.punish_epsilon:
            self.punish_epsilon = rewards
        # Calculate profit or loss of trade
        self.account_value(close_price, realizedPNL)
        #
        print(
            f'action = {action}, since last action = {i - self.last_action_time}, holding ={self.num_contracts}, '
            f'state = {state}, previous price = {previous_row["close"]}, current price = {row["close"]}, At open '
            f'price  = {self.priceAtStart}, unrealized pnl = {self.unrealized_profit_loss}, rewards = {rewards}, toal '
            f'invalid actions percentage = {self.totInvalidPerc}%')

        self.last_action_time = action_time
        return rewards

    def account_value(self, close_price, realizedPNL):
        # Calculate profit or loss of trade
        unrealized_profit_loss = self.num_contracts * (close_price - self.priceAtStart) * 20

        # Update realized PNL
        self.realized_profit_loss += realizedPNL

        # Update performance metrics
        self.profit = self.realized_profit_loss + unrealized_profit_loss
        self.unrealized_profit_loss = unrealized_profit_loss

        self.max_profit = max(self.max_profit, self.realized_profit_loss)
        self.max_loss = min(self.max_loss, self.realized_profit_loss)
        self.total_value = self.capital + self.profit

    def _get_kelly_criterion(self, win_loss_ratio, avg_profit):
        """Calculate number of contracts to trade using Kelly Criterion"""
        f = win_loss_ratio - 1
        if avg_profit == 0:
            avg_profit = 1
        b = avg_profit / 20
        kelly_criterion = f / b
        return kelly_criterion

    def calculate_rewards(self, state, price, previous_price, action, iteration, last_action_time,
                          realized_profit_loss, previous_action):
        time_since_last_action = iteration - last_action_time

        if state == 'invalid' or (action == previous_action and action != 2):
            return self.punish_epsilon  # severe punishment for invalid action

        # Encourage the agent to start a long or short trade
        if action == 0:  # start long trade
            return 50 if time_since_last_action >= 5 else -5000  # discourage repeated actions within 5 minutes
        elif action == 1:  # start short trade
            return 50 if time_since_last_action >= 5 else -5000  # discourage repeated actions within 5 minutes

        # Reward the agent for holding a position if it's profitable, discourage if it's not
        if action == 2:  # hold trade
            if state == "hold":
                return -50  # punishment for holding without any positions
            elif state == "long":
                return (price - previous_price) * (1 + time_since_last_action / 10) * self.num_contracts * 2
            elif state == "short":
                return (price - previous_price) * (1 + time_since_last_action / 10) * self.num_contracts * 2

        # Encourage the agent to close a trade when it's profitable and discourage taking the same action repeatedly
        if state == "close" and (action == 3 or action == 4):  # close long trade or close short trade
            return realized_profit_loss * (1 + time_since_last_action) / 100

        return self.punish_epsilon  # severe punishment for invalid action


class Market:
    """Class for handling market data"""

    def __init__(self, proxy=None, ibkr=False):

        self.scaler = MinMaxScaler(feature_range=(0, 4))
        self.trader = Trader()

        if ibkr:
            if proxy == None:
                self.ib = IB()

            else:
                self.ib = proxy

            self.checkIfReconnect()
            self.df = self.update_data(True)

    def checkIfReconnect(self):
        if not self.ib.isConnected() or not self.ib.client.isConnected():
            self.ib.disconnect()
            self.ib.connect('127.0.0.1', 7496, np.random.randint(0, 1000))

    def download_data(self, symbol='NQ', exchange='CME', currency='USD', length=100):
        self.contract = ContFuture(symbol, exchange)
        self.ib.qualifyContracts(self.contract)

        # Download historical data using reqHistoricalData
        self.bars = self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr=f'{length} S',
            barSizeSetting='5 secs', whatToShow='TRADES',
            useRTH=False
        )
        # Create a DataFrame from the downloaded data
        df = util.df(self.bars)
        # df = self.get_analysis(df)
        df.reset_index(inplace=True, drop=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def update_data(self, ibkr=False):

        if ibkr:
            df = self.download_data()
        else:
            df = self.load_data()
        df['contract'] = 0
        df['done'] = False

        self.df = df
        self.scaler.fit(self.df.values)

    def get_state(self, i=-1, numContracts=0):
        """Method for getting current state of market"""
        self.df.iloc[i, -2] = numContracts
        self.df["done"] = i + 2 >= len(self.df)
        state = self.df.iloc[i, :].values.reshape(1, -1)
        state = self.scaler.transform(state)
        return state

    def load_data(self, file=r'NQ_data.csv'):
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        df.dropna(inplace=True)
        return df

    def get_df(self):
        """Method for returning the DataFrame of market data"""
        return self.df

    def disconnect(self):
        """Method for disconnect"""
        self.ib.disconnect()

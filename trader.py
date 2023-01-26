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
        self.unreliazed_profit_loss = 0
        self.penalty = 1000
        self.expon = 1.05
        self.hold_penalty = -100
        self.last_action_time = 0

    def _get_kelly_criterion(self, win_loss_ratio, avg_profit):
        """Calculate the optimal number of contracts to trade using Kelly Criterion"""
        return win_loss_ratio - (1 - win_loss_ratio) / (avg_profit / 20)

    def trade(self, action, row, previous_row, i):
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
                Action = "close short and start long position"
                realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
                self.num_contracts = 0
                self.cash = abs(self.total_value) + realizedPNL
                if realizedPNL > 0:
                    self.num_wins += 1
                else:
                    self.num_losses += 1
            Action = "Long"
            self.num_contracts = int(self.cash / close_price / 20)
            self.cash = self.cash - abs(self.num_contracts * close_price * 20)
            self.priceAtStart = close_price
            self.num_trades += 1
        elif self.num_contracts >= 0 and action == 1:
            action_time = i
            if self.num_contracts > 0:
                self.priceAtClose = close_price
                Action = "close long and start short position"
                realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
                self.num_contracts = 0
                self.cash = abs(self.total_value) + realizedPNL
                if realizedPNL > 0:
                    self.num_wins += 1
                else:
                    self.num_losses += 1
            Action = "short"
            self.num_contracts = -1 * int(self.cash / close_price / 20)
            self.cash = abs(-1 * self.cash - self.num_contracts * close_price * 20)
            self.priceAtStart = close_price
            self.num_trades += 1
        elif self.num_contracts > 0 and action == 3:
            self.priceAtClose = close_price
            Action = "close"
            realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
            self.num_contracts = 0
            self.cash = abs(self.total_value) + realizedPNL
            if realizedPNL > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
        elif self.num_contracts < 0 and action == 4:
            self.priceAtClose = close_price
            Action = "close"
            realizedPNL = (self.priceAtStart - self.priceAtClose) * 20 * self.num_contracts
            self.num_contracts = 0
            self.cash = abs(self.total_value) + realizedPNL
            if realizedPNL > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
        elif self.num_contracts == 0 and action == 2:
            Action = "hold"
        elif action == 2:
            Action = "long" if self.num_contracts > 0 else "short"
        else:
            # action is "invalid"
            print(
                f'action = {action}, since last action = {i - self.last_action_time}, holding ={self.num_contracts}, '
                f'Action = invalid, rewards = -200')
            return -200  # punishment for invalid action

        rewards = self.calculate_rewards(Action, row['close'], previous_row['close'], action, i, self.last_action_time)
        print(
            f'action = {action}, since last action = {i - self.last_action_time}, holding ={self.num_contracts}, '
            f'Action = {Action}, rewards = {rewards}')

        # Calculate profit or loss of trade
        self.account_value(close_price, realizedPNL)

        self.last_action_time = action_time
        return rewards

    def account_value(self, close_price, realizedPNL):
        # Calculate profit or loss of trade
        unreliazed_profit_loss = self.num_contracts * (close_price - self.priceAtStart) * 20

        # Update realized PNL
        if realizedPNL != 0:
            self.realized_profit_loss += realizedPNL

        # Update performance metrics
        self.profit = unreliazed_profit_loss + self.realized_profit_loss
        self.unreliazed_profit_loss = unreliazed_profit_loss

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

    def calculate_rewards(self, state, price, previous_price, action, iteration, last_action_time):
        time_since_last_action = iteration - last_action_time
        if action == 0:  # start long trade
            if state == "long" and time_since_last_action < 3:  # last action was also long trade and taken less than 5 minutes ago
                return -100  # penalty for repeated action
            else:
                return 100  # increased reward for starting a trade
        elif action == 1:  # start short trade
            if state == "short" and time_since_last_action < 3:  # last action was also short trade and taken less than 5 minutes ago
                return -100  # penalty for repeated action
            else:
                return 100
        elif action == 2:  # hold trade
            if state == "hold":  # not holding any position
                return -25  # punishment for holding without any positions
            elif state == "long":
                if price > previous_price:
                    return 100  # increased reward for profitable trade
                else:
                    return -50
            elif state == "short":
                if price < previous_price:
                    return 100
                else:
                    return -50
        elif action == 3:  # close long trade
            if state == "close" and time_since_last_action > 3:  # last action was also long trade and taken less than 5 minutes ago
                return 50
            else:
                return -25  # punishment for frequent closing actions
        elif action == 4:  # close short trade
            if state == "close" and time_since_last_action > 3:  # last action was also short trade and taken less than 5 minutes ago
                return 50
            else:
                return -25  # punishment for frequent closing actions


class Market:
    """Class for handling market data"""

    def __init__(self, trader, ibkr=False, history_length=1, contract=None):
        # Connect to IB gateway
        if ibkr:
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7496, clientId=np.random.randint(0, 1000))

            # Store the history length as an instance variable
            self.history_length = history_length

            # Get the contract and data
            self.contract = contract
            self.ib.qualifyContracts(self.contract)
        self.scaler = MinMaxScaler(feature_range=(0, 4))
        self.trader = trader

    def download_data(self):
        self.contract = ContFuture('NQ', 'CME')
        self.ib.qualifyContracts(self.contract)
        # Download historical data using reqHistoricalData
        self.bars = self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr=f'{self.history_length} D',
            barSizeSetting='5 mins', whatToShow='TRADES',
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
        self.scaler.fit(df.values)

    def get_state(self, i=0):
        """Method for getting current state of market"""
        df = self.df.copy()
        df.iloc[i, -2] = self.trader.num_contracts
        df["done"] = i + 2 >= len(df)
        state = df.iloc[i, :].values.reshape(1, -1)

        state = self.scaler.transform(state)
        return state

    def load_data(self, file=r'NQ_data.csv'):
        df = pd.read_csv(file)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df

    def get_df(self):
        """Method for returning the DataFrame of market data"""
        return self.df

    def disconnect(self):
        """Method for disconnect"""
        self.ib.disconnect()

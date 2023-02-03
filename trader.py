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
        self.punish_epsilon = 0.1

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
            Action = "invalid"

        rewards = self.calculate_rewards(Action, row['close'], previous_row['close'], action, i, self.last_action_time,
                                         self.realized_profit_loss)

        # Calculate profit or loss of trade
        self.account_value(close_price, realizedPNL)
        #
        # print(
        #     f'action = {action}, since last action = {i - self.last_action_time}, holding ={self.num_contracts}, '
        #     f'Action = {Action}, unrealized pnl = {self.unreliazed_profit_loss}, rewards = {rewards}')

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
        self.unreliazed_profit_loss = 0.0 if unreliazed_profit_loss == 0.0 else 0.0 if unreliazed_profit_loss == -0.0 else unreliazed_profit_loss

        self.max_profit = max(self.max_profit, self.realized_profit_loss)
        self.max_loss = min(self.max_loss, self.realized_profit_loss)
        self.total_value = self.capital + self.profit

    def calculate_rewards(self, state, price, previous_price, action, iteration, last_action_time,
                          realized_profit_loss):
        time_since_last_action = iteration - last_action_time
        if state == 'invalid':
            self.punish_epsilon += 0.1
            return -2 - self.punish_epsilon  # punishment for invalid action
        elif action == 0:  # start long trade
            if time_since_last_action < 2:  # last action was also
                # long trade and taken less than 5 minutes ago
                return -1  # penalty for repeated action
            else:
                self.punish_epsilon = 0.1
                return 2  # increased reward for starting a trade
        elif action == 1:  # start short trade
            if time_since_last_action < 2:  # last action was also
                # short trade and taken less than 5 minutes ago
                return -1  # penalty for repeated action
            else:
                self.punish_epsilon = 0.1
                return 2
        elif action == 2:  # hold trade
            if state == "hold":  # not holding any position
                return -0.1 + self.punish_epsilon  # punishment for holding without any positions
            elif state == "long":
                # print(price, previous_price, price-previous_price, price-previous_price < 0, -1 if price - previous_price < 0 else 1)
                return -1 if price - previous_price < 0 else 1
            elif state == "short":
                # print(price, previous_price, previous_price - price, previous_price - price < 0, -1 if previous_price - price < 0 else 1)
                return -1 if previous_price - price < 0 else 1
        elif action == 3:  # close long trade
            if state == "close":
                self.punish_epsilon =0.1
                return 1
        elif action == 4:  # close short trade
            if state == "close":
                self.punish_epsilon = 0.1
                return 1
        else:
            return -2 - self.punish_epsilon  # punishment for invalid action


# self=self.market
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
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.dropna(inplace=True)
        return df

    def get_df(self):
        """Method for returning the DataFrame of market data"""
        return self.df

    def disconnect(self):
        """Method for disconnect"""
        self.ib.disconnect()

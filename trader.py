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

    def _get_kelly_criterion(self, win_loss_ratio, avg_profit):
        """Calculate the optimal number of contracts to trade using Kelly Criterion"""
        return win_loss_ratio - (1 - win_loss_ratio) / (avg_profit / 20)

    def trade(self, action, market, i, row, previous_row):
        """Execute a trade based on the current market state and the output of the model"""
        realizedPNL = 0
        # Get close price from Market object
        close_price = row['close']
        previous_close_price = previous_row['close']

        # Calculate number of contracts to trade using fixed-fractional method
        if self.num_contracts == 0 and action == 0:
            self.num_contracts = int(self.cash / close_price / 20)
            self.cash = self.cash - abs(self.num_contracts * close_price * 20)
            self.priceAtStart = close_price
            self.num_trades += 1
            Action = "buy"
        elif self.num_contracts == 0 and action == 1:
            self.num_contracts = -1 * int(self.cash / close_price / 20)
            self.cash = abs(-1 * self.cash - self.num_contracts * close_price * 20)
            self.priceAtStart = close_price
            self.num_trades += 1
            Action = "sell"
        elif self.num_contracts > 0 and action == 3:
            self.priceAtClose = close_price
            Action = "sell to close Longs"
            # print(f"*******Trade {self.num_trades} got {round((self.priceAtClose-self.priceAtStart)*20* self.num_contracts,2)} return*******")
            # print(f"********************************** Account value is {self.total_value +  round((self.priceAtClose-self.priceAtStart)*20* self.num_contracts,2)} **********************************")
            realizedPNL = (self.priceAtClose - self.priceAtStart) * 20 * self.num_contracts
            self.num_contracts = 0
            self.cash = abs(self.total_value) + realizedPNL
            if realizedPNL > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
            self.priceAtStart = 0
        elif self.num_contracts < 0 and action == 4:
            self.priceAtClose = close_price
            Action = "buy to close Shorts"
            # print(f"*******Trade {self.num_trades} got {round((self.priceAtStart-self.priceAtClose)*20*self.num_contracts,2)} return*******")
            # print(f"********************************** Account value is {self.total_value +  round((self.priceAtStart-self.priceAtClose)*20*self.num_contracts,2)} **********************************")
            realizedPNL = (self.priceAtStart - self.priceAtClose) * 20 * self.num_contracts
            self.num_contracts = 0
            self.cash = abs(self.total_value) + realizedPNL
            if realizedPNL > 0:
                self.num_wins += 1
            else:
                self.num_losses += 1
            self.priceAtStart = 0
        else:
            action = 10
            Action = "hold"

        # Calculate profit or loss of trade
        unreliazed_profit_loss = self.num_contracts * (close_price - self.priceAtStart) * 20

        if action == 2 and self.num_contracts == 0:
            unreliazed_profit_loss -= 1
        if action == 10:
            unreliazed_profit_loss -= 100

        if realizedPNL != 0:
            self.realized_profit_loss += realizedPNL

        # Update performance metrics
        self.profit = unreliazed_profit_loss + self.realized_profit_loss
        self.unreliazed_profit_loss = unreliazed_profit_loss

        self.max_profit = max(self.max_profit, self.realized_profit_loss)
        self.max_loss = min(self.max_loss, self.realized_profit_loss)
        self.total_value = self.capital + self.profit

        # print(f'Trade: {self.num_trades}, last close price: {close_price}, previous close price: {previous_close_price}, Action: {Action}, prices differences: { close_price - previous_close_price}; Unrealized Profit/Loss: {round(self.unreliazed_profit_loss,2)}, Realized PNL: {round(self.realized_profit_loss,2)}, Total PNL: {round(self.realized_profit_loss+self.unreliazed_profit_loss,2)}, Account NQ contracts holding = {self.num_contracts}, Account cash balance = {round(self.cash,2)}, Account total value = {round(self.total_value,2)} \n')

        return self.profit

    def _get_kelly_criterion(self, win_loss_ratio, avg_profit):
        """Calculate number of contracts to trade using Kelly Criterion"""
        f = win_loss_ratio - 1
        if avg_profit == 0:
            avg_profit = 1
        b = avg_profit / 20
        kelly_criterion = f / b
        return kelly_criterion




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
        self.df = df
        self.data = self.scaler.fit_transform(df)

    def get_state(self, i=0):
        """Method for getting current state of market"""
        state = self.data[i]
        contracts_holding = self.trader.num_contracts
        state[-1] = contracts_holding
        np.append(state, i + 2 >= len(self.data))
        return np.expand_dims(state, 0)

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
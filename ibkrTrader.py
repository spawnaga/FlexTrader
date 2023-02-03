# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:23:19 2023

@author: alial
"""
import numpy as np
from ib_insync import *
from trader import Trader, Market
<<<<<<< HEAD
from agentsEpsGreed import MultiTask
=======
from agents import MultiTask
>>>>>>> 97b72eb88e95ec18daaf333344a9334aa757c634

util.startLoop()


# self=ibkr=ibkrTrade()
class ibkrTrade:
    def __init__(self, symbol="NQ", exchange="CME", currency="USD", task='policy_gradient', action_size=5, length=1):
        # Initialize an instance of the Interactive Brokers API
        self.ib = IB()
        # Store the specified task
        self.task = task
        # Set trade profit/loss to 0
        self.tradePNL = 0
        # Initialize a future contract with the specified symbol, exchange, and currency
        self.contract = ContFuture(symbol, exchange, currency)
        # Check if the API is connected, if not, try to reconnect
        self.checkIfReconnect()
        # Qualify the contract to ensure it is valid
        self.ib.qualifyContracts(self.contract)
        # Request market data for the contract
        self.contractPrice = self.ib.reqMktData(self.contract, '', False, False, None)
        # Set the position to 0
        self.position = 0
        # Set the average cost to 0
        self.avgcost = 0
        # Initialize a `Trader` object
        self.trader = Trader()
        # Initialize a `Market` object with the IB API instance and set the `ibkr` attribute to `True`
        self.market = Market(proxy=self.ib, ibkr=True)
        # Update the market data
        self.market.update_data(True)
        # Get the latest market state
        state = self.market.get_state(i=-1)
        # Get the state size, which is the number of columns in the state
        state_size = state.shape[1]
        # Set the action to 2 (hold)
        self.action = 2
        # Initialize a `MultiTask` agent with the specified task, action size, state size, and job
        self.agent = MultiTask(task=task, action_size=action_size, state_size=state_size, job='test')
        # Request historical data for the contract with the specified length and bar size
        self.history = self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr=f'{length} D',
            barSizeSetting='5 secs', whatToShow='TRADES',
            useRTH=False, keepUpToDate=True,
            timeout=10)

    def connect(self, host='127.0.0.1', port=7496):
        """Connect to Interactive Brokers Trader Workstation (TWS) using the specified host and port."""
        self.ib.connect(host, port, np.random.randint(0, 1000))

    def checkIfReconnect(self):
        """Check if the connection to Interactive Brokers TWS is active, and if not, reconnect."""
        if not self.ib.isConnected() or not self.ib.client.isConnected():
            self.ib.disconnect()
            self.connect()

    def disconnect(self):
        """Disconnect from Interactive Brokers TWS."""
        self.ib.disconnect()

    def contract_position(self, event=None):
        # Get a list of all open positions
        positions = self.ib.positions()

        # Check if there are any open positions
        if positions != []:
            # Check if the open position is for the specified contract
            contractPosition = [position for position in positions if position.contract.symbol == self.contract.symbol]
            # Set the contract for the specified open position
            self.contract = contractPosition[0].contract
            # Set the number of shares for the specified open position
            self.position = contractPosition[0].position
            # Set the average cost per share for the specified open position
            self.avgcost = np.abs(contractPosition[0].avgCost / 20)

        # If there are no open positions
        else:
            # Set the position to 0
            self.position = 0
            # Set the average cost per share to 0
            self.avgcost = 0

    def trade(self, contract, hasNewBar=None):
        # Print the last known price of the contract
        print(f'price = {self.contractPrice.last}')

        # If a new bar has been created
        if hasNewBar:
            # If the position is greater than 0, calculate the trade profit/loss for a long position
            if self.position > 0:
                self.tradePNL = (self.contractPrice.ask - self.avgcost)
                print(f'long position of {self.contract.symbol} cost = {round(self.avgcost, 2)}')
                print(f'Trade profit/loss = {round(self.tradePNL * self.position * 20, 2)}')

            # If the position is less than 0, calculate the trade profit/loss for a short position
            elif self.position < 0:
                self.tradePNL = (self.avgcost - self.contractPrice.bid)
                print(f'short position of {self.contract.symbol} cost = {round(self.avgcost, 2)}')
                print(f'Total profit/loss = {round(self.tradePNL * abs(self.position) * 20, 2)}')

            # Get the next action based on the current state
            action = self.strategy()

            # If the action is 0 and the position is 0, start a long position
            if action == 0 and self.position == 0:
                print(f'price = {self.contractPrice.last}, action = start long')
                self.ordering("BUY", self.contract, 2, self.contractPrice)

            # If the action is 1 and the position is 0, start a short position
            elif action == 1 and self.position == 0:
                print(f'price = {self.contractPrice.last}, action = start short')
                self.ordering("SELL", self.contract, 2, self.contractPrice)

            # If the action is 2, hold the current position
            elif action == 2:
                print(f'price = {self.contractPrice.last}, action = hold')

            # If the action is 3 and the position is greater than 0, close the long position
            elif action == 3 and self.position > 0:
                print(f'price = {self.contractPrice.last}, action = close long')
                self.ordering("SELL", self.contract, self.position, self.contractPrice)

            # If the action is 4 and the position is less than 0, close the short position
            elif action == 4 and self.position < 0:
                print(f'price = {self.contractPrice.last}, action = close short')
                self.ordering("BUY", self.contract, abs(self.position), self.contractPrice)

    def strategy(self):
        # First, update the position of the contract
        self.contract_position()
        # Then update the market data
        self.market.update_data(True)
        # Get the state of the market
        state = self.market.get_state(i=-1, numContracts=self.position)
        # Get the action from the agent based on the current state and the task
        return self.agent.act(state=state, task=self.task, job='test')

    def ordering(self, action, contract, quantity, price):
        # function to place orders based on given action, contract, quantity, and price
        if len(self.ib.reqAllOpenOrders()) > 0:
            print('open order exist')
            self.ib.reqGlobalCancel()
            return
        print(f'{action} {contract.symbol}')

        self.contract.exchange = 'CME'
        order_ = LimitOrder(action=action, totalQuantity=abs(quantity),
                            lmtPrice=price.ask + 0.5 if action == "BUY" else price.bid - 0.5)
        trade_status = self.ib.placeOrder(contract, order_)
        self.ib.sleep(10)  # waiting 10 secs
        if not trade_status.orderStatus.remaining == 0:
            self.ib.cancelOrder(order_)  # canceling order if not filled

            return
        else:
            print(trade_status.orderStatus.status)

            return

    def error(self, reqId=None, errorCode=None, errorString=None, contract=None):  # error handler
        print(errorCode, errorString)

        if errorCode in [2104, 2108, 2158, 10182, 1102, 2106, 2107]:
            self.ib.cancelHistoricalData(self.contract)
            del self.contract

            print('attempt to restart data check')
            if len(self.contract) == 0:
                print(self.contract)
                self.error()
                self.reqId = []
            else:
                self.ib.sleep(1)
                self.contract.updateEvent += self.trade
                self.trade(self.contract)

        elif errorCode == 201:
            self.contract_position()
        elif errorCode == 321:
            self.contract.exchange = "CME"

    def main(self):
        self.ib.positionEvent += self.contract_position
        self.ib.errorEvent += self.error
        self.history.updateEvent += self.trade
        self.ib.run()


if __name__ == "__main__":
    ibk = ibkrTrade(task='actor_critic')
    ibk.main()

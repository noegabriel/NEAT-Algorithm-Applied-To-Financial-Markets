# © Noé Gabriel 2022 All Rights Reserved
# This file contains all the functions to be call with an external app to interact with the market environment

from sqlite3 import Timestamp
from sklearn import preprocessing
from stockstats import StockDataFrame
from pandas_datareader import wb

import pandas_datareader as web
import pandas_datareader as pdr
import datetime as dt
import pandas as pd
import numpy as np

def createPlayer():
  class Player:
    pass
  player = Player()
  player.balance = 1000
  player.totalValue = player.balance
  player.portfolioAssets = []
  return player

def quotesDownloader(symbol, start, end):
  
  # Request data
  fred = pdr.DataReader(['GDP', 'UNRATE', 'DFF', 'CORESTICKM159SFRBATL', 'VIXCLS'], 'fred', start, end).fillna(method='ffill').dropna()
  dxy = pdr.DataReader('DX-Y.NYB', 'yahoo', start, end).fillna(method='ffill').dropna()
  asset = pdr.DataReader(symbol, 'yahoo', start, end).fillna(method='ffill').dropna()

  # Format data
  cpi = pd.DataFrame(fred.iloc[:,3]).rename(columns={'CORESTICKM159SFRBATL': 'CPI'}) # Consumer Price Index
  dff = pd.DataFrame(fred.iloc[:,2]) # Federal Funds Effective Rate
  dxy = pd.DataFrame(dxy.iloc[:,3]).rename_axis('DATE').rename(columns={'Close': 'DXY'}) # USD Currency Index
  unr = pd.DataFrame(fred.iloc[:,1]) # Unemployment Rate
  vix = pd.DataFrame(fred.iloc[:,4]).rename(columns={'VIXCLS': 'VIX'}) # Volatility Index

  # Merge data
  macroData = cpi.merge(dff, on='DATE').merge(unr, on='DATE').merge(dxy, on='DATE').merge(vix, on='DATE')
  technicalData = StockDataFrame.retype(asset)[['mfi', 'rsi', 'macd', 'atr_20']].fillna(method='ffill').dropna().rename_axis('DATE').rename(columns={'mfi': 'MFI', 'rsi': 'RSI', 'macd': 'MACD', 'atr_20': 'ATR'})
  data = technicalData.merge(macroData, on='DATE')

  # Normalize data
  inputs = preprocessing.MinMaxScaler().fit_transform(data.values.tolist()).tolist()
  
  return [asset.values.tolist(), inputs]

def placeAnOrder(symbol, quotes, index, player, quantity, operation):
  price = quotes[0][index][3]

  # Buy operation
  if operation == 'buy':
    # Calculate the cost and define the fees
    cost = price * quantity
    fee = 0.5
    # Check if there is enough money to buy
    if cost <= player.balance :
      # Remove the money of the balance
      player.balance = player.balance - cost - (fee * quantity)
      # Check if there is already someting in the portfolio
      if player.portfolioAssets == []:
        # Add the asset symbol and the quantity
        player.portfolioAssets.append([symbol, quantity])
      else:
        # Add the corresponding quantity
        player.portfolioAssets[0][1] = player.portfolioAssets[0][1] + quantity
    else:
      # Return false because buying is not possible
      return False

  # Sell operation
  if operation == "sell":
    # Calculate the value of the sell operation and define the fees
    cost = price * quantity
    fee = 0.5
    # Check if the portfolio of asset is not empty
    if player.portfolioAssets != []:
      # Check if we have enough pieces of the asset to sell it
      if quantity <= player.portfolioAssets[0][1]:
        # Remove the corresponding quantity of the asset we wan't to sell 
        player.portfolioAssets[0][1] = player.portfolioAssets[0][1] - quantity
        # Credit the balance at the current price
        player.balance = player.balance + cost - (fee * quantity)
        # Clean the array of the portfolio of assets
        if player.portfolioAssets[0][1] == 0:
          player.portfolioAssets = []
      else:
        return False   
    else:
      # Return false because selling is not possible
      return False
    
  # Check if there is assets in the portfolio
  if player.portfolioAssets == []:
    # If the portfolio is empty of assets, the value of the portfdolio will be the same as the balance amount
    player.totalValue = player.balance
  else:
    # If there is assets in the portfolio, the value of the portfolio will be equal to the balance and the quantity of the assets multiplied by it's price
    player.totalValue = player.balance + (player.portfolioAssets[0][1] * price)

  return True

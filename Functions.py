# © Noé Gabriel 2022 All Rights Reserved

import matplotlib.pyplot as plt
import pandas as pd

def visualizePnlEvolution(player):
    evolution = pd.DataFrame(player.pnlEvolution, columns = ['DATE', 'PNL']) #.to_excel('bestTrainingReturn.xls', index=False)
    evolution.plot(x="DATE", y="PNL", kind="line")
    plt.show()

def visualizePortfolioEvolution(player):
    evolution = pd.DataFrame(player.portfolioEvolution, columns = ['DATE', 'PORTFOLIO']) #.to_excel('bestTrainingReturn.xls', index=False)
    evolution.plot(x="DATE", y="PORTFOLIO", kind="line")
    plt.show()

def changeInPercentage(value1, value2):
    return ((value2 - value1)/value1)*100

def holdStrategyReturn(quotes):
    start_price = quotes[0][0][3]
    end_price = quotes[0][len(quotes[0])-1][3]
    return changeInPercentage(start_price, end_price)

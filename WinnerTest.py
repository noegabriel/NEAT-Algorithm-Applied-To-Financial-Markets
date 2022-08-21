# © Noé Gabriel 2022 All Rights Reserved
# This file contains all the functions to be call with an external app to interact with the market environment

from genericpath import exists
from EnvironmentFunctions import *
import neat
import pickle
import os

# Parameters
quantity = 1
symbol = 'SPY'
start_date = '2021-01-01'
end_date = '2022-01-01'

inputs = quotesDownloader(symbol, start_date, end_date)
config_file = os.path.join('config.txt')

with open('winner.pickle', 'rb') as f :
    winner = pickle.load(f)

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

def test():
    index = 0
    player = createPlayer()
    # Passing into each state
    while(index < len(inputs[1])):
        data = inputs[1][index]
        # Use the neural network with each inputs
        output = winner_net.activate(data)
        if output[0] < 0.1:
            if placeAnOrder(symbol, inputs, index, player, quantity, 'sell') == True:
                print('inputs      : ', str(data))
                print('output      : ', str(output))             
                print('sell order filled (', str(quantity), ' * ', str(symbol), ' @ ', str(inputs[0][index][3]), ' $')
                print('PnL : ', str(player.pnl))               
        if output[0] > 0.9:
            if placeAnOrder(symbol, inputs, index, player, quantity, 'buy') == True:
                print('inputs      : ', str(data))
                print('output      : ', str(output))
                print('sell order filled (', str(quantity), ' * ', str(symbol), ' @ ', str(inputs[0][index][3]), ' $')
                print('PnL : ', str(player.pnl))               
        index += 1
    return player.pnl

print('Winner PnL : ', test())

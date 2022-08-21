from genericpath import exists
from EnvironmentFunctions import *
import neat
import pickle
import os

# Parameters
quantity = 1
symbol = 'SPY'
start_date = '2000-01-01'
end_date = '2020-01-01'

inputs = quotesDownloader(symbol, start_date, end_date)

# Evaluate each genome (player)
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        print('---------------- Genome Id ', str(genome_id), ' ----------------')
        # Create the player in the environment
        player = createPlayer()
        # Initialize the fitness value (profit and loss)
        genome.fitness = player.pnl
        # Create a random neural network according to the previous genomes and the config file
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Passing into each state
        index = 0
        while(index < len(inputs[1])):
            data = inputs[1][index]
            # Use the neural network with each inputs
            output = net.activate(data)
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
                    print('buy order filled (', str(quantity), ' * ', str(symbol), ' @ ', str(inputs[0][index][3]), ' $')
                    print('PnL : ', str(player.pnl))
            index += 1
        # Give the fitness value (profit and loss) to the genome (player)
        genome.fitness = player.pnl

    print('-------------------------------- Next generation --------------------------------')

# This is the main function of the algorithm
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population according to the config parameters
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(4))
    winner = p.run(eval_genomes, 10)
    # Display the winning genome.
    print('Best genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    index = 0
    player = createPlayer()
    # Save the winner.
    with open('winner.pickle', 'wb') as f:
        pickle.dump(winner, f)
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
                print('buy order filled (', str(quantity), ' * ', str(symbol), ' @ ', str(inputs[0][index][3]), ' $')
                print('PnL : ', str(player.pnl))
        index += 1
    if exists('neat-checkpoint-4'):
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

config_path = os.path.join('config.txt')
run(config_path)

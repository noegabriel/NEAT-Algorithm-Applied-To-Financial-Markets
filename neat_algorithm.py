from genericpath import exists
import os
from attr import NOTHING
from matplotlib.pyplot import spy
import neat
import datetime
import pickle
from EnvironmentFunctions import *

symbol = "SPY"
generations = 10
quantity = 1
start_date = '1993-01-29'
end_date = str(datetime.date.today())
inputs = quotesDownloader(symbol, start_date, end_date)

# Load a genome
def replay_genome(config_path, genome_path):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    return genome

# Evaluate each genome (player)
def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        print('------ NEXT PLAYER ------ ')
        # Create the player in the environment
        player = createPlayer()

        # Initialize the fitness value (score)
        genome.fitness = 0

        # Create a random neural network according to the previous genomes and the config file
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        index = 0

        # Passing into each state (xi = xor_inputs = "the list of each states with their inputs")
        while(index < len(inputs)):
            data = inputs[index]
            # Use the neural network with each inputs
            output = net.activate(data)
            #print(output)

            if output[0] < 0.4:
                if placeAnOrder(symbol, inputs, index, player, quantity, 'sell') == True:
                    #print('sell order filled')
                    c = 1

            if output[0] > 0.6:
                if placeAnOrder(symbol, inputs, index, player, quantity, 'buy') == True:
                    #print('buy  order filled')
                    c = 1

            if output[0] <= 0.6 and output[0] >= 0.4:
                if placeAnOrder(symbol, inputs, index, player, quantity, 'hold') == True:
                    #print('hold order filled')
                    c = 1
            index = index + 1

            player.totalValue = player.balance + player.portfolioValue
            if player.totalValue > 1000:
                print(player.totalValue)

        # Give the fitness value (score) to the genome (player)
        genome.fitness = player.totalValue - 1000

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

    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('Best genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    index = 0
    player = createPlayer()

    pickle.dump(winner_net, open('NEAT-Algorithm-Applied-To-Financial-Markets/winner.pkl', 'wb'))

    # Passing into each state (xi = xor_inputs = "the list of each states with their inputs")
    while(index < len(inputs)):
        data = inputs[index]
        # Use the neural network with each inputs
        output = winner_net.activate(data)

        if output[0] < 0.4:
            if placeAnOrder(symbol, inputs, index, player, quantity, 'sell') == True:
                print('sell order filled')

        if output[0] > 0.6:
            if placeAnOrder(symbol, inputs, index, player, quantity, 'buy') == True:
                print('buy  order filled')

        if output[0] <= 0.6 and output[0] >= 0.4:
            if placeAnOrder(symbol, inputs, index, player, quantity, 'hold') == True:
                print('hold order filled')

        index = index + 1
        player.totalValue = player.balance + player.portfolioValue
        if player.totalValue > 1000:
            print(player.totalValue)

    if exists('neat-checkpoint-4'):
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, generations)

config_path = os.path.join('NEAT-Algorithm-Applied-To-Financial-Markets/config.txt')
run(config_path)

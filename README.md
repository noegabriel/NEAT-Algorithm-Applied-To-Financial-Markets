# NEAT Algorithm Applied To Financial Markets

In this repository is the code of the [NEAT algorithm](https://neat-python.readthedocs.io/en/latest/neat_overview.html) customed to act in a [financial market simmulation environnment](https://github.com/noegabriel/Financial-Market-Simmulation-Environnment) in Python.

### Librairies
```Python
from genericpath import exists
from EnvironmentFunctions import *
import neat, datetime, pickle, os
```

### Parameters
```Python
symbol = "SPY"
generations = 10
quantity = 1
start_date = '1993-01-29'
end_date = str(datetime.date.today())
```

### Inputs

```Python
# Function
inputs = quotesDownloader(symbol, start_date, end_date)

# Strucutre of the inputs
inputs = [[Open	 High	 Low	Close	 AdjClose	 Volume]
          [Open	 High	 Low	Close	 AdjClose	 Volume]
          [Open	 High	 Low	Close	 AdjClose	 Volume]
          ...
          [Open	 High	 Low	Close	 AdjClose	 Volume]
          [Open	 High	 Low	Close	 AdjClose	 Volume]
          [Open	 High	 Low	Close	 AdjClose	 Volume]]
```

### Function to evaluate each genome (player)

```Python
def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
    
        print('------ NEXT PLAYER ------ ')
        
        # Create the player in the environment
        player = createPlayer()

        # Initialize the fitness value (the score represents the profit made)
        genome.fitness = 0

        # Create a random neural network according to the previous genomes and the config file
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        index = 0

        # Passing into each state
        while(index < len(inputs)):
        
            data = inputs[index]
            
            # Use the neural network with each inputs
            output = net.activate(data)
            print(output)

            # Take an action in the environment according to the output
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

        # Give the fitness value (score) to the genome (player)
        genome.fitness = player.totalValue - 1000
```

### Run function

```Python
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

    pickle.dump(winner_net, open('winner.pkl', 'wb'))

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
```

### Code to make the algorithm working

```Python
config_path = os.path.join('config.txt')
run(config_path)
```

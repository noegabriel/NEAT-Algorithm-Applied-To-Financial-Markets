# NEAT Algorithm Applied To Financial Markets

© Noé Gabriel 2022 All Rights Reserved

In this repository is the code of the [NEAT algorithm](https://neat-python.readthedocs.io/en/latest/neat_overview.html) customed to act in a [financial market simmulation environnment](https://github.com/noegabriel/Financial-Market-Simmulation-Environnment) in Python.

### Librairies
```Python
from genericpath import exists
from EnvironmentFunctions import *
import neat, datetime, pickle, os, stockstats
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

For the inputs, we will use :

Technical indicators on the S&P500 : 
- Money Flow Index (MFI)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Average True Range (ATR)

U.S. macroeconomic indicators :
- Consumer Price Index (CPI)
- Federal Funds Effective Rate (DFF)
- Unemployment Rate (UNRATE)
- USD Currency Index (DXY)
- Volatility Index (VIX)

### Function to get the inputs

```Python
def quotesDownloader(symbol, start, end):
  
  fred = pdr.DataReader(['GDP', 'UNRATE', 'DFF', 'CORESTICKM159SFRBATL', 'VIXCLS'], 'fred', start, end).fillna(method='ffill').dropna()
  dxy = pdr.DataReader('DX-Y.NYB', 'yahoo', start, end).fillna(method='ffill').dropna()
  spy = StockDataFrame.retype(pdr.DataReader('spy', 'yahoo', start, end).fillna(method='ffill').dropna())

  cpi = pd.DataFrame(fred.iloc[:,3]).rename(columns={'CORESTICKM159SFRBATL': 'CPI'}) # Consumer Price Index
  dff = pd.DataFrame(fred.iloc[:,2]) # Federal Funds Effective Rate
  dxy = pd.DataFrame(dxy.iloc[:,3]).rename_axis('DATE').rename(columns={'Close': 'DXY'}) # USD Currency Index
  unr = pd.DataFrame(fred.iloc[:,1]) # Unemployment Rate
  vix = pd.DataFrame(fred.iloc[:,4]).rename(columns={'VIXCLS': 'VIX'}) # Volatility Index

  macroData = cpi.merge(dff, on='DATE').merge(unr, on='DATE').merge(dxy, on='DATE').merge(vix, on='DATE')
  technicalData = spy[['mfi', 'rsi', 'macd', 'atr_20']].fillna(method='ffill').dropna().rename_axis('DATE').rename(columns={'mfi': 'MFI', 'rsi': 'RSI', 'macd': 'MACD', 'atr_20': 'ATR'})
  
  inputs = preprocessing.MinMaxScaler().fit_transform(technicalData.values)
  
  return technicalData.merge(macroData, on='DATE')

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

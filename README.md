# NEAT Algorithm Applied To Financial Markets

© [Noé Gabriel](https://www.linkedin.com/in/no%C3%A9-gabriel-b8337820b/) 2022 All Rights Reserved

In this repository is the code of the [NEAT algorithm](https://neat-python.readthedocs.io/en/latest/neat_overview.html) customed to act in a [financial market simmulation environnment](https://github.com/noegabriel/Financial-Market-Simmulation-Environnment) in Python.

### Deep Reinforcement Learning

![image](https://user-images.githubusercontent.com/84172514/185416631-ecb297b4-36f3-4837-b0ac-2c09a968bf89.png)

### Parameters
```Python
quantity = 10
generations = 100
symbol = 'BK'
start_date = '2000-01-01'
end_date = '2015-01-01'
```

### Inputs

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

Call the function :
```Python
inputs = quotesDownloader(symbol, start_date, end_date)
```

### Function to get the inputs

```Python
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
  
  return [asset.values.tolist(), inputs, asset.index.tolist()]
```

### Function to evaluate each genome (player)

```Python
# Evaluate each genome (player)
def eval_genomes(genomes, config):

    for genome_id, genome in genomes:

        print('Genome Id : ', str(genome_id))
        
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
```

### Run function

```Python
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
        
        # Add the state of the pnl to the pnlEvolution variable
        date = inputs[2][index]
        player.pnlEvolution.append([date, player.pnl])

        # Add the state of the portfolio to the portfolioEvolution variable
        portfolioValue = player.pnl + 1000
        player.portfolioEvolution.append([date, portfolioValue])
        
        # Next index
        index += 1

    if exists('neat-checkpoint-4'):
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)
```

### Code to make the algorithm working

```Python
config_path = os.path.join('config.txt')
run(config_path)
```

# Financial Market Simmulation Environnment

© [Noé Gabriel](https://www.linkedin.com/in/no%C3%A9-gabriel-b8337820b/) 2022 All Rights Reserved

In this repository is the code for a financial market simulation environment to be used for a deep reinforcement learning algorithm in Python.

### Environment requests

- Create standard player
  `createPlayers()`
- Place market orders for a particular player
  `placeAnOrder(symbol, quotes, date, player, quantity, operation)`
- Get the inputs
  `quotesDownloader(symbol, start, end)`

Market operations are `buy` or `sell`

### Attributes of players
```Python
  player.balance = 1000
  player.portfolioAssets = []
  player.pnl = 0
  player.pnlEvolution= []
  player.portfolioEvolution = []
```

Function to place an order :
```Python
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
    player.pnl = player.balance - 1000
  else:
    # If there is assets in the portfolio, the value of the portfolio will be equal to the balance and the quantity of the assets multiplied by it's price
    player.pnl = player.balance + (player.portfolioAssets[0][1] * price) - 1000

  return True
```

Sample of functions call :

```Python
# Create a player
player = createPlayer()

# Download the data of the specific asset we want to trade
quotes = quotesDownloader('TSLA', '2010-01-01', '2020-01-01')

# Show the total value of the portfolio
print(player.totalValue)

# Place an BUY order for 5 shares of TSLA in 2010-06-29
placeAnOrder('TSLA', quotes, '2010-06-29', player, 5, 'buy')

# Place an SELL order for 5 shares of TSLA in 2018-02-01
placeAnOrder('TSLA', quotes, '2018-02-01', player1, 5, 'sell')

# Show the total value of the portfolio
print(player.totalValue)
```

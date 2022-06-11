# NEAT-trader-trainer


This simple example code, was developed to show how we can use a NEAT algorithm to train a trading AI.
To achieve this goal, this AI uses 22 input neurons and 11 hidden neurons and only one output neuron. 
The activation function is tanh. 
When the output neurons value over 0.5 and we don't have an open positions buy a bitcoin. 
Pupulation level: 100 trader
## buy parameters:
- Risk-Rewar: 1:1.5 
- Stop lost level: 2 times ATR (pos 14)
- Take profit level 3 times ATR

## Input parameters:
- last 10 close price
- last 10 SMA 20 / price ratio
- last SMA 200 / price ratio

Each run try to trade for 3 years on Bitcoin. When this period is over, the best trader mutating and creates a new population from the best trader genom. 

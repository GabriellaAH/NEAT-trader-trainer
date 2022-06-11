import neat
import pandas as pd
import os
from finta import TA


gen = 0
idx = 0
df = pd.read_csv('BTC_19_22_5m.csv', index_col="startTime")
df['ema200'] = TA.SMA(df, period=200)
df['ema20'] = TA.SMA(df, period=20)
df['atr'] = TA.ATR(df)
index_labels = df.index.tolist()

class Trade:
    def __init__(self) -> None:
        self.haveposition = False
        self.tp = 0
        self.sl = 0
        self.win = 0
        self.lost = 0
        self.num_of_trade = 0
        
    def buy(self, tp, sl) -> None:
        if not self.haveposition:
            self.haveposition = True
            self.tp = tp 
            self.sl = sl
            self.num_of_trade += 1
             
    def step(self) -> int:
        global df, index_labels, idx
        
        high = df.loc[index_labels[idx], 'high']
        low = df.loc[index_labels[idx], 'low']
        
        if not self.haveposition:
            return 0
        if self.sl > low:
            self.haveposition = False
            self.tp = 0
            self.sl = 0
            self.lost += 1
            return 0
        if self.tp < high:
            self.haveposition = False
            self.tp = 0
            self.sl = 0
            self.win += 1
            return 1
        return 0
        
    def winrate(self):
        if (self.win + self.lost) > 0:
            return self.win / (self.win + self.lost)
        return 0
        
def eval_genomes(genomes, config):
    """
    runs the simulation 
    """
    global gen, df, idx
    gen += 1


    nets = []
    trades = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        trades.append(Trade())
        ge.append(genome)


    run = True
    pos = 300
    
    while run and len(df)-10 > pos:
        pos += 1

        st_p = df.loc[index_labels[pos-20], 'open']
        atr = df.loc[index_labels[pos], 'atr']
        e200 = df.loc[index_labels[pos], 'ema200'] / df.loc[index_labels[pos], 'close'] 

        price = df.loc[index_labels[pos], 'close']
        sl = price - (2*atr)   
        tp = price + (3 * atr)            
        ttp = tp / st_p

        neurons = (ttp, e200, )
        for i in range (pos, pos-10, -1):
            if (df.loc[index_labels[i], 'high'] > df.loc[index_labels[i], 'ema20'] and \
                df.loc[index_labels[i], 'low'] > df.loc[index_labels[i], 'ema20']) or \
            (df.loc[index_labels[i], 'high'] < df.loc[index_labels[i], 'ema20'] and \
                df.loc[index_labels[i], 'low'] < df.loc[index_labels[i], 'ema20']):
                s = df.loc[index_labels[i], 'ema20'] / df.loc[index_labels[i], 'close']
            else:
                s = 0
            c = df.loc[index_labels[i], 'close']/st_p
            neurons = neurons + (s, c,)          

        idx = pos
        for x, trd in enumerate(trades):  
            i = trd.step()
            if trd.num_of_trade > 10:
                ge[x].fitness = (trd.win *  1.5) - trd.lost
            else:
                ge[x].fitness = 0

            
            output = nets[trades.index(trd)].activate(neurons)


            if output[0] > 0.5:  # tanh activation function result will be between -1 and 1. if over 0.5 buy
                trd.buy(tp, sl)

            if pos%1000 == 0:
                print('Pos: %s, agent: %s Fitness: %s, trades: %.2f, winrate: %.2f' % (pos, x, ge[x].fitness, trd.num_of_trade, trd.winrate()*100))
                
        if pos%5000 == 0:
            old_trades = trades
            for trd in old_trades:
                if trd.num_of_trade == 0:
                    ge[trades.index(trd)].fitness -= 100
                    nets.pop(trades.index(trd))
                    ge.pop(trades.index(trd))
                    trades.pop(trades.index(trd))



def run(config_file):
    """
    runs the NEAT algorithm to train a neural network 
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)        
        
            
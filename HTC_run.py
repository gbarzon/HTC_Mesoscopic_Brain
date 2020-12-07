from HTC import HTC
from dask.distributed import Client
import numpy as np
import time
import os
import itertools

if __name__ == '__main__':
    # Simulation parameters
    runs = 100
    net = 'random'
    #nets = ['random', 'small', 'barabasi']
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Create directory for results
    folder = 'results/'+net+'/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Start Dask client
    client = Client()
    start = time.time()
    
    # create fake HTC for getting W_mean
    W_means = np.zeros(len(ps))
    for i, p in enumerate(ps):
        tmp = HTC(net, dT=0.03, p=p)
        W_means[i] = tmp.W_mean
    
    # provare a salvare tutti gli HTC per le varie probabilità e far partire gather tutti insieme
    sims = list(itertools.product(ps, range(runs)))
    
    # Create computation graph
    mods = client.map(lambda x: HTC(net, dT=0.03, p=x[0], Id=x[1], W_mean=W_means[np.array(ps)==x[0]][0]), sims)
    processed = client.map(lambda obj: obj.simulate(cluster=True, runs=1) and obj.save(folder), mods)

    # Run the actual computation
    client.gather(processed)
    
    stop = time.time()
    print('Total execution time: '+str((stop-start)/60)+'mins')

    client.close()
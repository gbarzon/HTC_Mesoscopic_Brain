from HTC import HTC
from dask.distributed import Client
import time
import os
import itertools

# Simulation parameters

if __name__ == '__main__':
    runs = 100
    net = 'random'
    #nets = ['random', 'small', 'barabasi']
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    #ps = [0.1]
    folder = 'results/'+net+'/'
    # Create directory for results
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Start Dask client
    client = Client()

    start = time.time()
    
    # provare a salvare tutti gli HTC per le varie probabilità e far partire gather tutti insieme
    sims = list(itertools.product(ps, range(runs)))
    
    # Create computation graph
    mods = client.map(lambda x: HTC(net, dT=0.03, p=x[0], Id=x[1]), sims)
    processed = client.map(lambda obj: obj.parallel_simulation(cluster=False, runs=1) and obj.save('results/random/'), mods)

    # Run the actual computation
    client.gather(processed)
    
    stop = time.time()
    print('Total execution time: '+str((stop-start)/60)+'mins')

    client.close()
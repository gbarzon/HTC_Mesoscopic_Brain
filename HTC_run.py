from HTC import HTC
from HTC_mean import get_mean_HTC
from dask.distributed import Client
import numpy as np
import time
import os
import itertools

if __name__ == '__main__':
    
    nets = ['random', 'small', 'barabasi']
    
    # Simulation parameters
    N = 66
    runs = 100
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    ks = [10, 15, 20, 25, 30]
    
    # Start Dask client
    client = Client()
    start = time.time()
    
    for net in nets:

        # Create directory for results
        folder = 'results/'+net+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        # create fake HTC for getting W_mean
        W_means = np.zeros(len(ps))
        names = []
        
        # If random -> loop over ps
        if net == 'random':
            # Store W_mean
            for i, p in enumerate(ps):
                # Create fake HTC
                tmp = HTC(net, N=N, dT=0.1, p=p)
                W_means[i] = tmp.W_mean
                names.append(tmp.name.rsplit('_', 1)[0])
            
            del tmp
    
            # Get list of different HTC models
            sims = list(itertools.product(ps, range(runs)))
    
            # Init computation graph
            mods = client.map(lambda x: HTC(net, N=N dT=0.03, p=x[0], Id=x[1], W_mean=W_means[np.array(ps)==x[0]][0]), sims)
            
        # Else -> loop over ks
        else:
            # Store W_mean
            for i, k in enumerate(ks):
                # Create fake HTC
                tmp = HTC(net, N=N, dT=0.1, k=k, p=0.5)
                W_means[i] = tmp.W_mean
                names.append(tmp.name.rsplit('_', 1)[0])
            
            del tmp
            
            # Get list of different HTC models
            sims = list(itertools.product(ks, range(runs)))
    
            # Init computation graph
            mods = client.map(lambda x: HTC(net, N=N, dT=0.03, p=0.5 k=x[0], Id=x[1], W_mean=W_means[np.array(ks)==x[0]][0]), sims)
        
        # Complete computation graph
        processed = client.map(lambda obj: obj.simulate(folder, cluster=True, dinamical=True, runs=1), mods)
        # Run the actual computation
        client.gather(processed)
    
        # Get mean object and clean folder
        for name in names:
            get_mean_HTC(folder, name, runs)
    
    stop = time.time()
    print('Total execution time: '+str((stop-start)/60)+'mins')

    client.close()

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
    dT = 0.03
    runs = 100
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    
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
                
        # Store W_mean
        for i, p in enumerate(ps):
            # Create fake HTC
            if net == 'random':
                tmp = HTC(net, N=N, dT=dT, p=p)
            else:
                tmp = HTC(net, N=N, dT=dT, k=int(p*N), p=0.5)
            W_means[i] = tmp.W_mean
            names.append(tmp.name.rsplit('_', 1)[0])
            
        del tmp
    
        # Get list of different HTC models
        sims = list(itertools.product(ps, range(runs)))
    
        # Init computation graph
        if net == 'random':
            mods = client.map(lambda x: HTC(net, N=N, dT=dT, p=x[0], Id=x[1], W_mean=W_means[np.array(ps)==x[0]][0]), sims)
        else:
            mods = client.map(lambda x: HTC(net, N=N, dT=dT, p=0.5, k=int(x[0]*N), Id=x[1], W_mean=W_means[np.array(ps)==x[0]][0]), sims)
        
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

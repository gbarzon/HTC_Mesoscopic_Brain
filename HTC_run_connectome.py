from HTC import HTC
from dask.distributed import Client
import time
import os
import itertools

if __name__ == '__main__':
    
    # Create directory for results
    folder = 'results/connectome/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    start = time.time()
    
    tmp = HTC('connectome', dT=0.01, Nstim=50)
    tmp.verbose=True
    tmp.simulate(folder, cluster=True, dinamical=True)
    
    stop = time.time()
    print('Total execution time: '+str((stop-start)/60)+'mins')
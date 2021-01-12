from HTC import HTC
import time
import os

if __name__ == '__main__':
    
    # Create directory for results
    folder = 'results/connectome_numba/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    start = time.time()
    
    tmp = HTC('connectome', dT=0.05, Nstim=50)
    tmp.verbose=True
    tmp.simulate(folder)
    
    stop = time.time()
    print('Total execution time: '+str((stop-start)/60)+'mins')
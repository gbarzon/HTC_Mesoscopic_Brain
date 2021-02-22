from HTC import HTC
from HTC_mean import get_mean_HTC
from dask.distributed import Client
import numpy as np
import time
import os
import itertools

if __name__ == '__main__':
    
    nets = ['random', 'small', 'barabasi']
    #nets = ['random']
    
    # Simulation parameters
    Ns = [int(5e2), int(1e3), int(2e3), int(5e3)]
    dT = 0.03
    runs = 50
    #ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    #ks = np.array([5,15,25,40,60])
    k = 15
    
    # Start Dask client
    client = Client()
    start = time.time()
    
    for net in nets:

        # Create directory for results
        folder = 'finite_size/'+net+'_'+str(k)+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        # create fake HTC for getting W_mean
        W_means = np.zeros(len(Ns))
        names = []
                
        # Store W_mean
        for i, N in enumerate(Ns):
            # Create fake HTC
            if net == 'random':
                tmp = HTC(net, N=N, dT=dT, p=k/N)
            elif net == 'small':
                tmp = HTC(net, N=N, dT=dT, k=k, p=0.5)
            elif net == 'barabasi':
                m = int( ( (2*N-1) - np.sqrt((2*N-1)**2 - 4*N*k) ) / 2 )
                tmp = HTC(net, N=N, dT=dT, k=m, p=0.5)
                
            W_means[i] = tmp.W_mean
            names.append(tmp.name.rsplit('_', 1)[0])
            
        del tmp
    
        # Get list of different HTC models
        sims = list(itertools.product(Ns, range(runs)))
    
        # Init computation graph
        if net == 'random':
            mods = client.map(lambda x: HTC(net, N=x[0], dT=dT, p=k/x[0], Id=x[1], W_mean=W_means[np.array(Ns)==x[0]][0]), sims)
        elif net == 'small':
            mods = client.map(lambda x: HTC(net, N=x[0], dT=dT, p=0.5, k=k, Id=x[1], W_mean=W_means[np.array(Ns)==x[0]][0]), sims)
        elif net == 'barabasi':
            mods = client.map(lambda x: HTC(net, N=x[0], dT=dT, p=0.5, k=int(( (2*x[0]-1) - np.sqrt((2*x[0]-1)**2 - 4*x[0]*k) ) / 2), Id=x[1], W_mean=W_means[np.array(Ns)==x[0]][0]), sims)
        
        # Complete computation graph
        processed = client.map(lambda obj: obj.simulate(folder, runs=1), mods)
        # Run the actual computation
        client.gather(processed)
        
        
        # Get mean object and clean folder
        #for name in names:
        #    get_mean_HTC(folder, name, runs)
    
    stop = time.time()
    print('Total execution time: '+str((stop-start)/60)+'mins')

    client.close()

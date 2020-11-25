from HTC import HTC
from dask.distributed import Client
import time
import os

# Simulation parameters

runs = 100
net = 'random'
#nets = ['random', 'small', 'barabasi']
#ps = [0.1, 0.2, 0.3, 0.4, 0.5]
ps = 0.1

folder = 'results/'+net+'/'
# Create directory for results
if not os.path.exists(folder):
    os.makedirs(folder)

# Start Dask client
client = Client()

start = time.time()

for p in ps:
    print('p='+str(p))
    
    mods = client.map(lambda i: HTC(net, dT=0.03, p=p, Id=i), range(runs))
    processed = client.map(lambda obj: obj.parallel_simulation(cluster=True, runs=1) and obj.save('results/random/'), mods)

    # Run the actual computation
    client.gather(processed)
    
stop = time.time()
print('Total execution time: '+str((stop-start)/60)+'mins')

client.close()

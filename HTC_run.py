from HTC import HTC
import time

start = time.time()

a = HTC('connectome')

a.simulate(cluster=True)
a.save()

stop = time.time()
print('Total execution time: '+str(stop-start))

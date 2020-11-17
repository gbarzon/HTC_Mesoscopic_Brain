from HTC import HTC
import time

start = time.time()

a = HTC('connectome', dT=0.01)

a.simulate(cluster=False, runs=500)
a.save()

stop = time.time()
print('Total execution time: '+str(stop-start))

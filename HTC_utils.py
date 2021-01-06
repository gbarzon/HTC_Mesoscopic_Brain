import numpy as np
from numba import jit, prange
from scipy.signal import periodogram
import igraph as gf
from collections import Counter


# ---------- GENERAL FUNCTIONS ----------
def power_law(a, b, g, size):
    ''' Power-law gen for pdf(x) prop to x^{g} for a<=x<=b '''
    if g == -1:
        raise Exception('g must be different from -1')
    r = np.random.random(size=size)
    ag, bg = a**(g+1), b**(g+1)

    return (ag + (bg - ag)*r)**(1./(g+1))

def normalize(W):
    ''' Normalize each entry in a matrix by the sum of its row'''
    return W / np.sum(W, axis=1)[:,None]

@jit(nopython=True)
def hline_intersection(x1, y1, x2, y2, y_star):
    '''
    Get intersection btw lines through two points and hline
    '''
    if np.sum((y1-y2)==0)>0:
        print('hline_intersection: the y\'s are equal')
    return (x1 - x2) / (y1 - y2) * (y_star - y2) + x2


# ---------- PDF/POWER SPECTRUM IO HANDLING ----------
def reshape_pdf(pdf):
    ''' Reshape list of counter to sorted array '''
    pdf = [np.array(list(i.items())).T for i in pdf]
    pdf = [i[:, i[0].argsort()] for i in pdf]
    
    return pdf

def write_lists(lst, lst_norm, fname):
    ''' Write lists of numpy vectors to .txt file'''
    with open(fname, 'w') as outfile:
        for x in lst:
            np.savetxt(outfile, x)
            outfile.write('\n')
        for x in lst_norm:
            np.savetxt(outfile, x)
            outfile.write('\n')


def read_lists(fname):
    ''' Read lists of numpy vectors from .txt file'''
    text_file = open(fname, 'r')
    lines = text_file.read().split('\n\n')
    del lines[-1]

    lines = [i.split('\n') for i in lines]
    lst = []

    for i in lines:
        lst.append( np.array([j.split(' ') for j in i]).astype(float) )
    return lst[:len(lst)//2], lst[len(lst)//2:]


# ---------- HTC SIMULATION ----------
@jit(nopython=True)
def init_state(N, runs, fract):
    '''
    Initialize the state of the system
    fract: fraction of initial acrive neurons
    ''' 
    n_act = np.ceil(fract * N)     # number of initial active neurons
    n_act = int(n_act)
        
    # create vector with n_act 1's, the rest half 0's and half -1's
    ss = np.zeros(N)
    ss[:n_act] = 1.
    ss[-(N-n_act)//2:] = -1.
    
    # create shuffled array
    states = np.zeros((runs,N))
    for i in prange(runs):
        states[i] = np.random.choice(ss, len(ss), replace=False)
        
    return states
    
    
@jit(nopython=True)
def update_state_single(S, W, T, r1, r2, aval=None):
    '''
    Update state of the system according to HTC model
    '''
        
    probs = np.random.random(len(S))                 # generate probabilities
    s = (S==1).astype(np.float64)                    # get active nodes
    pA = ( r1 + (1.-r1) * ( (W@s)>T ) )              # prob. to become active

    # update state vector
    newS = ( (S==0)*(probs<pA)                       # I->A
         + (S==1)*-1                                 # A->R
         + (S==-1)*(probs>r2)*-1 )                   # R->I (remain R with prob 1-r2)
        
    return newS

@jit(nopython=True, parallel=True)
def update_state(S, W, T, r1, r2, aval=None):
    '''
    Update state of each runs
    '''
    
    runs = S.shape[0]
    newS = np.zeros((S.shape[0], S.shape[1]))
    
    # Loop over runs
    for i in prange(runs):
        tmpS = update_state_single(S[i], W, T, r1, r2, aval=None)
        newS[i] = tmpS
        
    return (newS, (newS==1).astype(np.int64))

@jit(nopython=True)
def stimulated_activity(W, runs, steps, r1, r2):
    '''
    Simulate activity with external stimulus
    '''
    N = W.shape[0]
    S = init_state(N, runs, fract)
    At = np.zeros((runs, steps))
                
    # Loop over time steps
    for t in prange(steps):
        # update state vector
        S, s = update_state(S, W, T, r1=r1, r2=r2)
        # compute average activity
        At[:,t] = np.mean(s, axis=1)
    # end loop over time steps
    return np.mean(At)
    

# ---------- HTC OBSERVABLES ----------
@jit(nopython=True)
def fisher_information(mat):
    '''
    Return the average Fisher information btw each run.
    Fisher Information = mean covariance btw N time series.
    mat -> (runs)x(steps)x(nodes) matrix
    '''    
    runs, steps, N = mat.shape
    
    fisher = 0
    
    # Loop over runs
    for i in prange(runs):
        # compute NxN covariance matrix
        Cij = np.cov(mat[i], rowvar=False)
        # get only upper-triangular values
        mask = np.triu( np.ones((N,N)), k=1)
        Cij = (Cij * mask).flatten()
        Cij = Cij[Cij != 0]
        
        fisher += np.mean(Cij)
        
    return fisher / runs


def entropy(data):
    '''
    Compute the ensamble entropy and its std
    '''
    p = np.mean(data, axis=1) # node activation average
    ent = -( p*np.log2(p) + (1.-p)*np.log2(1.-p) ) # node entropy
    ent[np.isnan(ent)] = 0.
    ent[np.logical_not(np.isfinite(ent))] = 0.
    ent = np.mean(ent, axis=1) # run entropy

    # return ensemble mean/std
    return np.mean(ent), np.std(ent)


def power_spectrum(data, dt=1.):
    '''
    Compute the power spectrum of a collection of time series
    using a periodogram.
    - Input: data(runs,steps), timestep
    - Output: frequencies, power spectrum
    '''
    # compute the power spectrum for each run
    return periodogram(data, scaling='spectrum', fs=1./dt)

def avg_pow_spec(data):
    '''
    Compute the average power spectrum of a collection of time series
    using a periodogram.
    - Input: data(runs,steps), timestep
    - Output: power spectrum
    '''
    spectr = np.array(list(map(power_spectrum, data)))
    
    #freq = spectr[0,0]
    spectr = np.mean(spectr[:,1], axis=0)
    
    return spectr


@jit(nopython=True)
def interevent(data):
    '''
    Compute the time btw two consecutives activations
    of a node
    '''
    runs, steps, nodes = data.shape
    
    dt_cumulative = np.array([0.0])
    
    for i in prange(runs): 
        ind = np.where(data[i]==1)          # get active times
    
        dt = ind[1][1:]-ind[1][:-1]         # compute dt
        dt = dt[ind[0][1:]-ind[0][:-1]==0]  # discard dt from different trials
        dt_cumulative = np.hstack((dt_cumulative, dt))
    
    return np.delete(dt_cumulative, 0)


# ---------- CLUSTER ANALYSIS ----------

@jit(nopython=True)
def myCount(cluster):
    '''
    Count the occurrence of unique labels
    '''
    # Get unique labels
    keys = np.unique(cluster)
    N = len(keys)
    sizes = np.zeros(N)
    
    # Count label occurrences
    for i in prange(N):
        sizes[i] = np.count_nonzero(cluster==keys[i])
        
    return -np.sort(-sizes)


@jit(nopython=True)
def myConnectedComponents(W, s):
    '''
    Get the size of the connected components
    '''
    N = W.shape[0]
    
    # mask adjacency matrix with active nodes
    mask = (W * s).T * s
    
    # Init all nodes belongs to fake cluster
    cluster = np.zeros(N)-1
    
    # Loop over nodes
    for i in range(N):
        # If not already assigned, create new cluster
        if cluster[i] == -1:
            cluster[i] = i
        for j in range(i+1,N):
            if cluster[j] == -1 and mask[i,j]>0:
                cluster[j] = cluster[i]
                
    # Return occurences
    return myCount(cluster)


@jit(nopython=True, parallel=False)
def compute_clusters(W, s):
    '''
    Cluster size analysis for each run
    '''
    runs, N = s.shape
    
    # Initialize arrays
    clusters = np.zeros((runs, N))
    S1 = np.zeros(runs)
    S2 = np.zeros(runs)
    
    # Loop over runs
    for i in prange(runs):
        sizes = myConnectedComponents(W, s[i])
        ss = len(sizes)
        S1[i] = sizes[0]
        if ss>1:
            S2[i] = sizes[1]
        clusters[i,:ss] = sizes
    
    clusters = clusters.flatten() # flatten
    clusters = clusters[clusters>0] # remove fake zeros
    
    return np.mean(S1), np.mean(S2), clusters.flatten()


# ---------- AVALANCHES ----------
@jit(nopython=True)
def get_intersection(arr, y_star):
    '''
    Get all the intersection btw array and a hline
    '''
    # Increase intersection
    start = np.where( (arr[:-1]<y_star) * (y_star<arr[1:]) )[0]
    # Decrease intersection
    stop = np.where( (arr[:-1]>y_star) * (y_star>arr[1:]) )[0]
    
    # Check if empty
    if len(start)<1 or len(stop)<1:
        return np.empty(0), np.empty(0)
    
    # Check that first start is smaller than first stop
    if start[0]>stop[0]:
        stop = np.delete(stop, 0)
    # Check that last stop is bigger than last start
    if stop[-1]<start[-1]:
        start = np.delete(start, -1)
        
    # Check if empty
    if len(start)<1 or len(stop)<1:
        return np.empty(0), np.empty(0)
    
    # Get intersection with hline
    start = hline_intersection(start, arr[start], start+1, arr[start+1], y_star)
    stop = hline_intersection(stop, arr[stop], stop+1, arr[stop+1], y_star)
    
    return start, stop


@jit(nopython=True)
def get_avalanches(arr, y_star):
    '''
    Get sizes and lifetimes of avalanches in a single run
    '''
    T = len(arr)
    t = np.arange(T)
    
    # Get start and stop of avalanches
    start, stop = get_intersection(arr, y_star)
    
    # Check if empty
    if len(start)<1 or len(stop)<1:
        return np.empty(0), np.empty(0)
    
    # Compute avalanche time
    dt = stop - start
    
    # Compute avalanche size as activity area
    I = np.zeros(len(start))
    
    for i in range(len(start)):
        # Get point btw each start and stop
        t_in = t[(t>start[i])*(t<stop[i])]
        y_in = arr[t_in]
    
        # Append start and stop
        t_in = np.hstack(( np.array([start[i]]), t_in, np.array([stop[i]]) ))
        y_in = np.hstack(( np.array([y_star]), y_in, np.array([y_star]) ))
    
        # Integrate spline
        I[i] = np.trapz(y=y_in, x=t_in)
        
    return I, dt


@jit(nopython=True)
def avalanches(data, threshold=1., Nbins=50):
    '''
    Return avalanches histogram from series of runs.
    '''
    av_size = np.array([0.0])
    av_time = np.array([0.0])
    
    runs = data.shape[0]
    
    for i in prange(runs):
        tmp_size, tmp_time = get_avalanches(data[i], np.mean(data[i]) + threshold * np.std(data[i]))
        av_size = np.hstack((av_size, tmp_size))
        av_time = np.hstack((av_time, tmp_time))
        
    # Remove fake 0 as first element
    av_size = np.delete(av_size, 0)
    av_time = np.delete(av_time, 0)
    
    # Get histograms to compress data
    hist_size = np.histogram(av_size, bins = Nbins)
    hist_time = np.histogram(av_time, bins = Nbins)
            
    # Convert bin edges to bin center
    hist_size = np.vstack( ( (hist_size[1][1:] + hist_size[1][:-1])/2, hist_size[0] ) )
    hist_time = np.vstack( ( (hist_time[1][1:] + hist_time[1][:-1])/2, hist_time[0] ) )
    
    return hist_size, hist_time


# ---------- ANALYSIS POST-SIMULATION ----------
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def get_dynamical_range(mod):
    '''
    Compute the dynamical range
    '''
    delta = np.zeros(len(mod.Trange))
    delta_norm = np.zeros(len(mod.Trange))
    
    for i in range(len(mod.Exc)):
        # Original matrix
        
        # Get A90 and A10
        Amax, Amin = np.max(mod.Exc[i]), np.min(mod.Exc[i])
        A10, A90 = (Amax-Amin)*0.15 + Amin, (Amax-Amin)*0.85 + Amin
        # Get corresponent index
        s10 = np.where( (A10 > mod.Exc[i][:-1])*(A10 < mod.Exc[i][1:]) )[0][-1]
        s90 = np.where( (A90 > mod.Exc[i][:-1])*(A90 < mod.Exc[i][1:]) )[0][0]
        # Get the s value by linear interpolation
        s10 = (mod.stimuli[s10]-mod.stimuli[s10+1]) / (mod.Exc[i][s10]-mod.Exc[i][s10+1]) * (A10 - mod.Exc[i][s10+1]) + mod.stimuli[s10+1]
        s90 = (mod.stimuli[s90]-mod.stimuli[s90+1]) / (mod.Exc[i][s90]-mod.Exc[i][s90+1]) * (A90 - mod.Exc[i][s90+1]) + mod.stimuli[s90+1]
        
        # Dynamical range
        delta[i] = 10*np.log10(s90 / s10)
    
        # Original matrix
        
        # Get A90 and A10
        Amax, Amin = np.max(mod.Exc_norm[i]), np.min(mod.Exc_norm[i])
        A10, A90 = (Amax-Amin)*0.15 + Amin, (Amax-Amin)*0.85 + Amin
        # Get corresponent index
        s10 = np.where( (A10 > mod.Exc_norm[i][:-1])*(A10 < mod.Exc_norm[i][1:]) )[0][-1]
        s90 = np.where( (A90 > mod.Exc_norm[i][:-1])*(A90 < mod.Exc_norm[i][1:]) )[0][0]
        # Get the s value by linear interpolation
        s10 = (mod.stimuli[s10]-mod.stimuli[s10+1]) / (mod.Exc_norm[i][s10]-mod.Exc_norm[i][s10+1]) * (A10 - mod.Exc_norm[i][s10+1]) + mod.stimuli[s10+1]
        s90 = (mod.stimuli[s90]-mod.stimuli[s90+1]) / (mod.Exc_norm[i][s90]-mod.Exc_norm[i][s90+1]) * (A90 - mod.Exc_norm[i][s90+1]) + mod.stimuli[s90+1]
        
        # Dynamical range
        delta_norm[i] = 10*np.log10(s90 / s10)
        
    return delta, delta_norm

def get_Tc(mod):
    '''
    Get critical parameter, define as the maximum of S2
    '''
    Tc = mod.Trange[np.argmax(mod.S2)] * mod.W_mean
    Tc_norm = mod.Trange[np.argmax(mod.S2_norm)]
    
    return Tc, Tc_norm
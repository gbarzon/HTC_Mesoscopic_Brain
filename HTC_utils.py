import numpy as np
from scipy.signal import periodogram
import igraph as gf
from collections import Counter

# GENERAL FUNCTION
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


# PDF/POWER SPECTRUM IO HANDLING
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
    text_file = open(fname, "r")
    lines = text_file.read().split('\n\n')
    del lines[-1]

    lines = [i.split('\n') for i in lines]
    lst = []

    for i in lines:
        lst.append( np.array([j.split(' ') for j in i]).astype(float) )
    return lst[:len(lst)//2], lst[len(lst)//2:]


# HTC SIMULATION
def init_state(N, runs, fract):
    '''
    Initialize the state of the system
    fract: fraction of initial acrive neurons
    '''
    from math import ceil
        
    n_act = ceil(fract * N)     # number of initial active neurons
        
    # create vector with n_act 1's, the rest half 0's and half -1's
    ss = np.zeros(N)
    ss[:n_act] = 1.
    ss[-(N-n_act)//2:] = -1.
        
    # return shuffled array
    return np.array([np.random.choice(ss, len(ss), replace=False) for _ in range(runs)])
    
    
def update_state(S, W, T, r1, r2):
    '''
    Update state of the system according to HTC model
    '''
        
    probs = np.random.rand(S.shape[0], S.shape[1])   # generate probabilities
    s = (S==1).astype(int)                           # get active nodes
    pA = r1 + (1.-r1) * ( (W@s.T)>T )                # prob. to become active

    # update state vector
    newS = ( (S==0)*(probs<pA.T)                     # I->A
         + (S==1)*-1                                 # A->R
         + (S==-1)*(probs>r2)*-1 )                   # R->I (remain R with prob 1-r2)
        
    return (newS, (newS==1).astype(int) )


def compute_clusters(W, sj):
    '''
    Compute cluster analysis
    '''
    # mask adjacency matrix with active nodes
    mask = (W * sj).T * sj
    # create igraph object
    graph = gf.Graph.Adjacency( (mask > 0).tolist())
    # compute connected components occurrence
    counts = np.array(graph.clusters().sizes())
    counts = -np.sort(-counts)
        
    # return (biggest cluster, second biggest cluster, clusters occurrence)
    return (counts[0], counts[1], Counter(counts))


def stimulated_activity(W, runs, steps, r1, r2):
    '''
    Simulate activity with external stimulus
    '''
    N = W.shape[0]
    S = init_state(N, runs, fract)
    At = np.zeros((runs, steps))
                
    # Loop over time steps
    for t in range(steps):
        # update state vector
        S, s = update_state(S, W, T, r1=r1, r2=r2)
        # compute average activity
        At[:,t] = np.mean(s, axis=1)
    # end loop over time steps
    return np.mean(At)
    
    
def correlation(mat):
    '''
    Return the (mean and std) correlation btw N time series.
    mat -> MxN matrix where:
    - M is the legth of each time series
    - N is the number of different time series
    '''    
    N = mat.shape[1]
    
    Cij = np.corrcoef(mat, rowvar=False)        # compute NxN correlation matrix
    Cij = Cij[np.triu_indices(N, k=1)]          # get only upper-triangular values

    return ( np.mean(Cij), np.std(Cij) )


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
    - Output: frequencies, power spectrum
    '''
    spectr = np.array(list(map(power_spectrum, data)))
    
    freq = spectr[0,0]
    spectr = np.mean(spectr[:,1], axis=0)
    
    return np.array([freq, spectr])


def interevent(data):
    '''
    Compute the time btw two consecutives activations
    of a node
    '''
    steps, nodes = data.shape
    
    # check nodes with only O or 1 activation
    # -> set dt equal to length of simulation
    not_active = np.sum(np.sum(data, axis=1)<=1.)
    
    ind = np.where(data==1)             # get active times
    
    dt = ind[1][1:]-ind[1][:-1]         # compute dt
    dt = dt[ind[0][1:]-ind[0][:-1]==0]  # discard dt from different trials
    dt = np.append(dt, [steps]*not_active)
    
    return np.mean(dt), np.std(dt), Counter(dt)

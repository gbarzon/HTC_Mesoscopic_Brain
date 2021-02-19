import numpy as np
import networkx as nx

from tqdm.auto import tqdm
from IPython.display import clear_output
from pathlib import Path

from HTC_utils import *

#results_folder = 'results/'
delimiter = '_'

class HTC:
    
    def __init__(self, network, generator=True, W = None,
                 weights='power_law', N=66,
                 Tmin = 0.0, Tmax = 1.5, dT = 0.03, W_mean=None,
                 Nstim = 20, Id = 0,
                 **kwargs):
        '''
        Class initializer
        '''
        self.verbose = False
        
        self.Id = Id
        self.W = W
        self.network = network
        self.N = N
        self.weights = weights
        
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.dT = dT
        self.W_mean = W_mean
        self.Nstim = Nstim
        self.dt = 0.1
        
        # Unpack parameters
        self.unpack_parameters(**kwargs)
        
        # Create network topology and compute parameters
        if generator:
            self.generate_network()
        self.compute_parameters()
        
        print('CREATED ' + self.title + ', id=' + str(self.Id) + ' ...\n')
        
     
    @classmethod
    def loadFromName(cls, filename):
        name = filename.split('/')[-1]
        network = name.split(delimiter)[0]
        
        # Load weights matrix
        #W = np.loadtxt(filename+delimiter+'matrix.txt')
        
        # Unpack parameters from name
        N, k, p = [0, 0, 0]
        
        if network == 'connectome':
            network, N = name.split(delimiter)[:2]
        elif network == 'random':
            network, N, p = name.split(delimiter)[:3]
        elif network == 'small':
            network, N, k, p = name.split(delimiter)[:4]
        elif network == 'barabasi':
            network, N, k = name.split(delimiter)[:3]
        elif network == 'powerlaw':
            network, N, k, p = name.split(delimiter)[:4]
        
        Tmin, Tmax, dT, Nstim, W_mean, Id = map(float, name.split(delimiter)[-6:])
        Nstim, Id = int(Nstim), int(Id)
        
        # Create HTC object
        tmp = cls(network=network, N=int(N), k=int(k), p=float(p), Tmin=Tmin,
                  Tmax=Tmax, dT=dT, Nstim=Nstim, Id=Id, W_mean=W_mean, generator=False)
        
        # Load time series (if present)
        if Path(filename+delimiter+str('series.txt')).is_file():
            act = np.loadtxt(filename+delimiter+str('series.txt'))
            tmp.act, tmp.act_norm = act[:len(act)//2], act[len(act)//2:]
            del act
        
        # Load power spectrum
        if Path(filename+delimiter+str('spectrum.txt')).is_file():
            spectr = np.loadtxt(filename+delimiter+str('spectrum.txt'))
            tmp.spectr, tmp.spectr_norm = spectr[:len(spectr)//2], spectr[len(spectr)//2:]
            del spectr
        
        ### Load pdfs
        # Cluster
        if Path(filename+delimiter+str('pdf.txt')).is_file():
            tmp.pdf, tmp.pdf_norm = read_lists(filename+delimiter+str('pdf.txt'))
        # Interevent
        if Path(filename+delimiter+str('pdf_ev.txt')).is_file():
            tmp.pdf_ev, tmp.pdf_ev_norm = read_lists(filename+delimiter+'pdf_ev.txt')
        # Aval size
        if Path(filename+delimiter+str('pdf_aval_size.txt')).is_file():
            tmp.pdf_size, tmp.pdf_size_norm = read_lists(filename+delimiter+'pdf_aval_size.txt')
        # Aval time
        if Path(filename+delimiter+str('pdf_aval_time.txt')).is_file():
            tmp.pdf_time, tmp.pdf_time_norm = read_lists(filename+delimiter+'pdf_aval_time.txt')
        # Causal aval size
        if Path(filename+delimiter+str('pdf_aval_size_causal.txt')).is_file():
            tmp.pdf_size_causal, tmp.pdf_size_causal_norm = read_lists(filename+delimiter+'pdf_aval_size_causal.txt')
        # Causal aval time
        if Path(filename+delimiter+str('pdf_aval_time_causal.txt')).is_file():
            tmp.pdf_time_causal, tmp.pdf_time_causal_norm = read_lists(filename+delimiter+'pdf_aval_time_causal.txt')
        
        # Load stimulated activity (if present)
        if Path(filename+delimiter+str('stimulated.txt')).is_file():
            exc = np.loadtxt(filename+delimiter+str('stimulated.txt'))
            tmp.Exc, tmp.Exc_norm = exc[:len(exc)//2], exc[len(exc)//2:]
            del exc
        
        # Load activity and (if present) cluster size
        obs = np.loadtxt(filename+delimiter+str('observables.txt'))
        
        # Load observables
        tmp.A, tmp.sigmaA, tmp.Fisher, tmp.S1, tmp.S2, tmp.Smean, \
        tmp.A_norm, tmp.sigmaA_norm, tmp.Fisher_norm, tmp.S1_norm, tmp.S2_norm, tmp.Smean_norm = obs
        
        return tmp
    
    
    def unpack_parameters(self, **kwargs):
        '''
        Unpack parameters and create name/title
        '''
        
        self.title = 'Network=' + self.network + ', N=' + str(self.N)
        self.name = self.network + delimiter + str(self.N)
    
        if self.network == 'random':
            if not 'p' in kwargs.keys():
                raise Exception('Random network generator needs p')
            self.p = kwargs['p']
            self.title += ', p=' + str(self.p)
            self.name += delimiter + str(self.p)
            
        elif self.network == 'small':
            if not ('k' and 'p') in kwargs.keys():
                raise Exception('Small world generator needs k and p')
            self.k = kwargs['k']
            self.p = kwargs['p']
            self.title += ', k=' + str(self.k) + ', p=' + str(self.p)
            self.name += delimiter + str(self.k) + delimiter + str(self.p)
            
        elif self.network == 'barabasi':
            if not 'k' in kwargs.keys():
                raise Exception('Barabasi generator needs k')
            self.k = kwargs['k']
            self.title += ', k=' + str(self.k)
            self.name += delimiter + str(self.k)
            
        elif self.network == 'powerlaw':
            if not ('k' and 'p') in kwargs.keys():
                raise Exception('Powerlaw network generator needs k and p')
            self.k = kwargs['k']
            self.p = kwargs['p']
            self.title += ', k=' + str(self.k) + ', p=' + str(self.p)
            self.name += delimiter + str(self.k) + delimiter + str(self.p)
            
        # add Tmin, Tmax, dT to the name
        self.name += delimiter + str(self.Tmin) + delimiter + str(self.Tmax) \
                    + delimiter + str(self.dT) + delimiter + str(self.Nstim)
        
    
    def generate_network(self):
        '''
        Generate network topology and edge weights
        '''
        
        # Check if network read from .txt
        if not self.W is None:
            if not isinstance(self.W, (np.ndarray)):
                raise Exception('The type of the connectivity matrix W is wrong')
            return
        
        # Generate adjacency matrix
        if self.network == 'connectome':
            self.W = np.loadtxt('dati/RSN/RSN_matrix.txt')
            #self.W = np.loadtxt('dati/Hagmann/group_mean_connectivity_matrix.txt')
        else:
            if self.network == 'random':
                top = nx.erdos_renyi_graph(self.N, self.p)
            elif self.network == 'small':
                top = nx.watts_strogatz_graph(self.N, self.k, self.p)
            elif self.network == 'barabasi':
                top = nx.barabasi_albert_graph(self.N, self.k)
            elif self.network == 'powerlaw':
                top = nx.powerlaw_cluster_graph(self.N, self.k, self.p)
            else:
                raise Exception('Network not defined')
                
            top = nx.adjacency_matrix(top).toarray()
            
            # Generate weights from distribution
            if self.weights == 'unif':
                self.W = np.random.uniform(0., .5, size=(self.N, self.N))
            elif self.weights == 'power_law':
                self.W = power_law(5*10**-3, .5, g=-1.5, size=(self.N, self.N))
            elif self.weights == 'constant':
                self.W = np.ones( (self.N, self.N) )
            else:
                raise Exception('Incorrect weight distribution')
            
            # Symmetrize and mask with topology
            self.W = np.triu(self.W, 1)
            self.W = self.W + self.W.T
            self.W = self.W * top
            
        # Check if connected, otherwise try to generate again
        if not nx.is_connected(nx.from_numpy_matrix(self.W)):
            self.W = None
            print('WARNING: the network is not connected. Generate again.')
            self.generate_network()

    def compute_parameters(self):
        '''
        Compute transition rates and (theorical) critical temperature
        '''
        
        self.r1 = 2./self.N
        self.r2 = self.r1**(1./5.)
        self.Tc = self.r2 / (1. + 2.*self.r2)
        self.Trange = np.arange(self.Tmin, self.Tmax+self.dT, self.dT) * self.Tc
        self.stimuli = np.logspace(-5, 0, self.Nstim, endpoint=True)
        if self.W_mean == None:
            self.W_mean = round( np.mean(np.sum(self.W, axis=1)), 2 )
        self.name += delimiter + str(self.W_mean) + delimiter + str(self.Id)
    
    
    def simulate(self, results_folder,
                 cluster=True, dinamical=False, complete_simulation=False,
                 steps=6000, runs=100):
        '''
        Run simulation for both original and normalized matrices
        '''
        print('Start simulation for '+str(self.name))
        
        self.A, self.sigmaA, self.Fisher, self.spectr, self.act, self.pdf_ev, self.Exc, \
        self.pdf_size, self.pdf_time, self.pdf_size_causal, self.pdf_time_causal, \
        self.S1, self.S2, self.Smean, self.pdf = \
        self.run_model(self.W, cluster, dinamical, complete_simulation, steps, runs)
            
        self.A_norm, self.sigmaA_norm, self.Fisher_norm, self.spectr_norm, self.act_norm, self.pdf_ev_norm, self.Exc_norm, \
        self.pdf_size_norm, self.pdf_time_norm, self.pdf_size_causal_norm, self.pdf_time_causal_norm, \
        self.S1_norm, self.S2_norm, self.Smean_norm, self.pdf_norm = \
        self.run_model(normalize(self.W), cluster, dinamical, complete_simulation, steps, runs)
        
        print('End simulation for '+str(self.name))
        # Save results
        self.save(results_folder, cluster, dinamical, complete_simulation)
    
    
    def run_model(self, W, cluster, dinamical, complete_simulation, steps, runs, fract=0.1, steps_cluster=5):
        '''
        HTC model
        '''
                
        # treshold interval
        if np.mean(np.sum(W, axis=1)) == 1:
            W_mean = 1
        else:
            W_mean = self.W_mean

        Trange = self.Trange * W_mean
        
        # define empty matrix to store results
        A, sigma_A = [np.zeros(len(Trange)) for _ in range(2)]
        Fisher = np.zeros(len(Trange))
        
        pdf_ev = [Counter() for _ in range(len(Trange))]
        pdf_size = []
        pdf_time = []
        pdf_size_causal = []
        pdf_time_causal = []
        
        spectr = []
        act = np.zeros((len(Trange), steps))
        
        Exc = np.zeros((len(Trange),len(self.stimuli)))
        
        S1, S2, Smean = [np.zeros(len(Trange)) for _ in range(3)]
        pdf = [Counter() for _ in range(len(Trange))]

        if self.verbose:
            print(self.title)
            if W_mean==1:
                print('START SIMULATION WITH NORMALIZED MATRIX...')
            else:
                print('START SIMULATION WITH ORIGINAL MATRIX...')
            
        # LOOP OVER TEMPERATUREs
        for i,T in enumerate(Trange):
            if self.verbose:
                clear_output(wait=True)
                #print(self.title + '\n')
                print('\n'+str(i+1) + '/'+ str(len(Trange)) + ' - T = ' +  str(round(T/self.Tc/W_mean, 2)) + ' * Tc' )
                print('Simulating activity...')

            # MODEL INITIALIZATION
            # create empty array to store activity and cluster size over time
            Aij = np.zeros((runs, steps, self.N))
            #aval_ij = np.zeros((steps, runs, self.N))
            
            # init activity and avalanches
            S = init_state(self.N, runs, fract)
            #aval = (S==1) * np.arange(1,self.N+1)
            #aval = aval.astype(np.int32)

            if cluster:
                S1t = np.zeros(steps//steps_cluster)
                S2t = np.zeros(steps//steps_cluster)
                Smeant = np.zeros(steps//steps_cluster)

            # LOOP OVER TIME STEPS
            for t in ( tqdm(range(steps)) if self.verbose else range(steps)):
                # UPDATE STATE VECTOR
                #S, s, aval = update_state(S, W, T, self.r1, self.r2, aval, t, avalOn=True)
                #Aij[:,t] = s
                #aval_ij[t] = aval
                S, s = update_state(S, W, T, self.r1, self.r2)
                Aij[:,t] = s

                # COMPUTE CLUSTERS
                if cluster:
                    if t%steps_cluster == 0:
                        S1t[t//steps_cluster], S2t[t//steps_cluster], Smeant[t//steps_cluster], tmp_counts = compute_clusters(W, s)
                        pdf[i] += Counter(tmp_counts)
                 
            # clear tmp variables
            del S, s#, aval
            # END LOOP OVER TIME
            
            # COMPUTE AVERAGES
            # Activity
            if self.verbose: print('Computing activity...')
            At = np.mean(Aij, axis=2)    # node average <A(t)>
            A[i], sigma_A[i] = np.mean(At), np.mean( np.std(At, axis=1) )
            act[i] = At[0]
            
            # cluster
            if cluster:
                S1[i] = np.mean(S1t)
                S2[i] = np.mean(S2t)
                Smean[i] = np.mean(Smeant)
                
                del S1t, S2t, Smeant
            # END COMPUTE AVERAGES
            
            # FISHER INFORMATION
            if self.verbose: print('Computing Fisher information...')
            Fisher[i] = fisher_information(Aij)
            # END FISHER INFORMATION
            
            del Aij
            
            if complete_simulation:
                # INTER-EVENT TIME
                if self.verbose: print('Computing interevent time...')
                pdf_ev[i] = Counter(interevent(Aij))
            
                del Aij
            
                # POWER SPECTRUM
                if self.verbose: print('Computing power spectrum...')
                spectr.append(avg_pow_spec(At))
            
                # COMPUTE AVALANCHES
                
                if self.verbose: print('Computing avalanches...')
                sizes, times = get_avalanches_pdf(At)
                
                # Get histograms
                Nbins = 50
                hist_size = np.histogram(sizes,
                                         bins = np.logspace( np.log10(np.min(sizes)), np.log10(np.max(sizes)), Nbins),
                                         density=True)
                
                hist_time = np.histogram(times,
                                         bins = np.logspace( np.log10(np.min(times)), np.log10(np.max(times)), Nbins),
                                         density=True)
                
                # Reshape histograms
                center = 10 ** ( ( np.log10(hist_size[1][1:]) + np.log10(hist_size[1][:-1]) ) / 2 )
                hist_size = np.array(( center, hist_size[0] ))
                center = 10 ** ( ( np.log10(hist_time[1][1:]) + np.log10(hist_time[1][:-1]) ) / 2 )
                hist_time = np.array(( center, hist_time[0] ))
                pdf_size.append( hist_size )
                pdf_time.append( hist_time )
            
                del At
                # END COMPUTE AVALANCHES
            
                # COMPUTE CAUSAL AVALANCHES
                if self.verbose: print('Computing causal avalanches...')
                sizes, times = get_causal_avalanches_pdf(aval_ij)
                
                # Get histograms
                Nbins = 50
                hist_size = np.histogram(sizes,
                                         bins = np.logspace( np.log10(np.min(sizes)), np.log10(np.max(sizes)), Nbins),
                                         density=True)
                
                hist_time = np.histogram(times,
                                         bins = np.logspace( np.log10(np.min(times)), np.log10(np.max(times)), Nbins),
                                         density=True)
                
                # Reshape histograms
                center = 10 ** ( ( np.log10(hist_size[1][1:]) + np.log10(hist_size[1][:-1]) ) / 2 )
                hist_size = np.array(( center, hist_size[0] ))
                center = 10 ** ( ( np.log10(hist_time[1][1:]) + np.log10(hist_time[1][:-1]) ) / 2 )
                hist_time = np.array(( center, hist_time[0] ))
                pdf_size_causal.append( hist_size )
                pdf_time_causal.append( hist_time )
                
                del hist_size, hist_time, aval_ij   # clear tmp variables
                # END COMPUTE CAUSAL AVALANCHES
                
            # DYNAMICAL RANGE
            if dinamical:
                if self.verbose: print('\nSimulating dynamical range...')
                
                Exc[i] = stimulated(self.stimuli, self.N, W, T, self.r2, runs, fract, steps//10)
            # END DYNAMICAL RANGE
        
        # END LOOP OVER TEMPERATUREs
        if self.verbose:
            clear_output(wait=True)
            print(self.title + '\n')
            print('End simulating activity')
        
        if complete_simulation:
            # Reshape pdfs
            pdf_ev = reshape_pdf(pdf_ev)
            # Reshape spectrum
            spectr = np.vstack(spectr)

        # RETURN RESULTS
        if cluster:
            # Reshape cluster pdf
            pdf = reshape_pdf(pdf)
            
        return (A, sigma_A, Fisher, spectr, act, pdf_ev, Exc,
                pdf_size, pdf_time, pdf_size_causal, pdf_time_causal,
                S1/self.N, S2/self.N, Smean, pdf)
        
        
    def save(self, results_folder, cluster, dinamical, complete_simulation):
        '''
        Save output
        '''        
        
        filename = results_folder+self.name
        
        # Save weights matrix
        #np.savetxt(filename + delimiter + 'matrix.txt', self.W)
        
        # Save activity
        if self.network == 'connectome':
            np.savetxt(filename + delimiter + 'series.txt', np.vstack((self.act, self.act_norm)), fmt='%e')
        
        if complete_simulation:
            # Save power spectrum
            np.savetxt(filename + delimiter + 'spectrum.txt', np.vstack((self.spectr, self.spectr_norm)), fmt='%e')
            # Save interevent pdf
            write_lists(self.pdf_ev, self.pdf_ev_norm, filename + delimiter + 'pdf_ev.txt')
            # Save avalanches pdf
            write_lists(self.pdf_size, self.pdf_size_norm, filename + delimiter + 'pdf_aval_size.txt')
            write_lists(self.pdf_time, self.pdf_time_norm, filename + delimiter + 'pdf_aval_time.txt')
            # Save causal avalanches pdf
            write_lists(self.pdf_size_causal, self.pdf_size_causal_norm, filename + delimiter + 'pdf_aval_size_causal.txt')
            write_lists(self.pdf_time_causal, self.pdf_time_causal_norm, filename + delimiter + 'pdf_aval_time_causal.txt')
                    
        # Save stimulated activity
        if dinamical:
            np.savetxt(filename + delimiter + 'stimulated.txt', np.vstack((self.Exc, self.Exc_norm)), fmt='%e')
        
        np.savetxt(filename + delimiter + 'observables.txt',
                   (self.A, self.sigmaA, self.Fisher,
                    self.S1, self.S2, self.Smean,
                    self.A_norm, self.sigmaA_norm, self.Fisher_norm,
                    self.S1_norm, self.S2_norm, self.Smean_norm), fmt='%e')
        
        if cluster:
            write_lists(self.pdf, self.pdf_norm, filename + delimiter + 'pdf.txt')
        
        print('Saved '+str(self.name))

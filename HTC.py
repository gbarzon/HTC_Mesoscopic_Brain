import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components

from tqdm.auto import tqdm
from IPython.display import clear_output

from HTC_utils import *

results_folder = 'tmpresults/'
delimiter = '_'

class HTC:
    
    def __init__(self, network, W = None,
                 weights='power_law', N=66, 
                 Tmin = 0.0, Tmax = 1.5, dT = 0.03,
                 **kwargs):
        '''
        Class initializer
        '''
        
        self.W = W
        self.network = network
        self.N = N
        self.weights = weights
        
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.dT = dT
        self.dt = 0.1
        
        # Unpack parameters
        self.unpack_parameters(**kwargs)
        
        # Create network topology and compute parameters
        self.generate_network()
        self.compute_parameters()
        
        print('CREATED ' + self.title + ' ...\n')
        # Check if the network is connected
        if not nx.is_connected(nx.from_numpy_matrix(self.W)):
            print('WARNING: the network is not connected')
        
     
    @classmethod
    def loadFromName(cls, name):
        network = name.split(delimiter)[0]
        
        # Load weights matrix
        W = np.loadtxt(results_folder+name+delimiter+'matrix.txt')
        
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
        
        Tmin, Tmax, dT = map(float, name.split(delimiter)[-3:])
        
        # Create HTC object
        tmp = cls(network=network, W=W, N=int(N), k=int(k), p=float(p), Tmin=Tmin, Tmax=Tmax, dT=dT)
        
        # Load activity and (if present) cluster size
        obs = np.loadtxt(results_folder+name+delimiter+str('observables.txt'))
        
        # Load time series
        act = np.loadtxt(results_folder+name+delimiter+str('series.txt'))
        tmp.act, tmp.act_norm = act[:len(act)//2], act[len(act)//2:]
        
        # Load power spectrum
        text_file = open(results_folder+name+delimiter+str('spectrum.txt'), "r")
        lines = text_file.read().split('\n\n')
        del lines[-1]

        lines = [i.split('\n') for i in lines]
        spectr = []

        for i in lines:
            spectr.append( np.array([j.split(' ') for j in i]).astype(float) )
        tmp.spectr, tmp.spectr_norm = spectr[:len(spectr)//2], spectr[len(spectr)//2:]
        
        # Load observables
        if len(obs)==28:
            tmp.A, tmp.sigmaA, tmp.C, tmp.sigmaC, tmp.Ent, tmp.sigmaEnt,\
            tmp.Ev, tmp.sigmaEv, tmp.Tau, tmp.sigmaTau, tmp.Chi, tmp.sigmaChi,\
            tmp.S1, tmp.S2,\
            tmp.A_norm, tmp.sigmaA_norm, tmp.C_norm, tmp.sigmaC_norm, tmp.Ent_norm, tmp.sigmaEnt_norm,\
            tmp.Ev_norm, tmp.sigmaEv_norm, tmp.Tau_norm, tmp.sigmaTau_norm, tmp.Chi_norm, tmp.sigmaChi_norm,\
            tmp.S1_norm, tmp.S2_norm = obs
            
            # Unpack cluster distribution
            text_file = open(results_folder+name+delimiter+str('pdf.txt'), "r")
            lines = text_file.read().split('\n\n')
            del lines[-1]

            lines = [i.split('\n') for i in lines]
            pdf = []

            for i in lines:
                pdf.append( np.array([j.split(' ') for j in i]).astype(float).astype(int) )
            tmp.pdf, tmp.pdf_norm = pdf[:int(len(pdf)/2)], pdf[int(len(pdf)/2):]
        else:
            tmp.A, tmp.sigmaA, tmp.C, tmp.sigmaC, tmp.Ent, tmp.sigmaEnt,\
            tmp.Ev, tmp.sigmaEv, tmp.Tau, tmp.sigmaTau, tmp.Chi, tmp.sigmaChi,\
            tmp.A_norm, tmp.sigmaA_norm, tmp.C_norm, tmp.sigmaC_norm, tmp.Ent_norm, tmp.sigmaEnt_norm,\
            tmp.Ev_norm, tmp.sigmaEv_norm, tmp.Tau_norm, tmp.sigmaTau_norm, tmp.Chi_norm, tmp.sigmaChi_norm,\
            = obs
        
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
        self.name += delimiter + str(self.Tmin) + delimiter + str(self.Tmax) + delimiter + str(self.dT)
        
    
    def generate_network(self):
        '''
        Generate network topology and edge weights
        '''
        
        # Check if network read from .txt
        if not self.W is None:
            if not isinstance(self.W, (np.ndarray)):
                raise Exception('The type of the connectivity matrix W is wrong')
            # Create normalized matrix
            self.W_norm = normalize(self.W)
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
            else:
                raise Exception('Incorrect weight distribution')
            
            # Symmetrize and mask with topology
            self.W = np.triu(self.W, 1)
            self.W = self.W + self.W.T
            self.W = self.W * top
        
        # Create normalized matrix
        self.W_norm = normalize(self.W)
        
    
    def compute_parameters(self):
        '''
        Compute transition rates and (theorical) critical temperature
        '''
        
        self.r1 = 2./self.N
        self.r2 = self.r1**(1./5.)
        self.Tc = self.r2 / (1. + 2.*self.r2)
        self.Trange = np.arange(self.Tmin, self.Tmax+self.dT, self.dT) * self.Tc
    
    
    def simulate(self, cluster=False, steps=6000, runs=100, N_cluster=3000):
        '''
        Run simulation for both original and normalized matrices
        '''
        
        self.cluster = cluster
        
        if cluster:
            self.A, self.sigmaA, self.C, self.sigmaC, self.Ent, self.sigmaEnt,\
            self.Ev, self.sigmaEv, self.Tau, self.sigmaTau, self.Chi, self.sigmaChi,\
            self.spectr, self.act, self.S1, self.S2, self.pdf = \
            self.run_model(self.W, cluster, steps, runs, N_cluster)
            
            self.A_norm, self.sigmaA_norm, self.C_norm, self.sigmaC_norm, self.Ent_norm, self.sigmaEnt_norm,\
            self.Ev_norm, self.sigmaEv_norm, self.Tau_norm, self.sigmaTau_norm, self.Chi_norm, self.sigmaChi_norm,\
            self.spectr_norm, self.act_norm, self.S1_norm, self.S2_norm, self.pdf_norm = \
            self.run_model(self.W_norm, cluster, steps, runs, N_cluster)
        else:
            self.A, self.sigmaA, self.C, self.sigmaC, self.Ent, self.sigmaEnt,\
            self.Ev, self.sigmaEv, self.Tau, self.sigmaTau, self.Chi, self.sigmaChi,\
            self.spectr, self.act = self.run_model(self.W, cluster, steps, runs, N_cluster)
            
            self.A_norm, self.sigmaA_norm, self.C_norm, self.sigmaC_norm, self.Ent_norm, self.sigmaEnt_norm,\
            self.Ev_norm, self.sigmaEv_norm, self.Tau_norm, self.sigmaTau_norm, self.Chi_norm, self.sigmaChi_norm,\
            self.spectr_norm, self.act_norm = self.run_model(self.W_norm, cluster, steps, runs, N_cluster)
            
    
    def run_model(self, W, cluster, steps, runs, N_cluster, fract=0.):
        '''
        HTC model
        '''
        
        dt_cluster = int(steps/N_cluster)
        r1_old = self.r1
        
        # treshold interval
        W_mean = np.mean(np.sum(W, axis=1))
        Trange = self.Trange * W_mean
        
        # define empty matrix to store results
        A, sigma_A = [np.zeros(len(Trange)) for _ in range(2)]
        C, sigma_C = [np.zeros(len(Trange)) for _ in range(2)]
        ent, sigma_ent = [np.zeros(len(Trange)) for _ in range(2)]
        ev, sigma_ev = [np.zeros(len(Trange)) for _ in range(2)]
        tau, sigma_tau = [np.zeros(len(Trange)) for _ in range(2)]
        chi, sigma_chi = [np.zeros(len(Trange)) for _ in range(2)]
        spectr = []
        act = np.zeros((len(Trange), steps))
        
        if cluster:
            S1, S2 = [np.zeros(len(Trange)) for _ in range(2)]
            pdf = [Counter() for _ in range(len(Trange))]

        print(self.title)
        if W_mean==1.:
            print('START SIMULATION WITH NORMALIZED MATRIX...')
        else:
            print('START SIMULATION WITH ORIGINAL MATRIX...')
            
        # LOOP OVER TEMPERATUREs
        for i,T in enumerate(Trange):
            clear_output(wait=True)
            #print(self.title + '\n')
            print('\n'+str(i+1) + '/'+ str(len(Trange)) + ' - T = ' +  str(round(T/self.Tc/W_mean, 2)) + ' * Tc' )
            print('Simulating activity...')

            # MODEL INITIALIZATION
            S = self.init_state(runs, fract)

            # create empty array to store activity and cluster size over time
            Aij = np.zeros((runs, steps, self.N))

            if cluster:
                S1t = np.zeros((runs, N_cluster))
                S2t = np.zeros((runs, N_cluster))

            # LOOP OVER TIME STEPS
            for t in tqdm(range(steps)):
                # UPDATE STATE VECTOR
                S, s = self.update_state(S, W, T)
                Aij[:,t,:] = s

                # COMPUTE CLUSTERS
                if cluster and (not t%dt_cluster):
                    tempT = t//dt_cluster
                    for j in range(runs):
                        S1t[j,tempT], S2t[j,tempT], tmp_counts = compute_clusters(W, s[j])
                        pdf[i] += tmp_counts
            # END LOOP OVER TIME

            # COMPUTE AVERAGES
            # activity
            print('Computing activity...')
            At = np.mean(Aij, axis=2)    # node average <A(t)>
            A[i], sigma_A[i] = np.mean(At), np.mean( np.std(At, axis=1) )
            act[i] = At[0]
            
            # correlation
            print('Computing correlation...')
            tmpCij = np.array(list(map(correlation, Aij)))
            C[i], sigma_C[i] = np.mean(tmpCij[:,0]), np.mean(tmpCij[:,1])
            
            # entropy
            print('Computing entropy...')
            ent[i], sigma_ent[i] = entropy(Aij)
            
            # inter-event time
            print('Computing interevent time...')
            tmpEv = np.array(list(map(interevent, Aij)))
            ev[i], sigma_ev[i] = np.mean(tmpEv[:,0]), np.mean(tmpEv[:,1])
            
            # power spectrum
            print('Computing power spectrum...')
            spectr.append(avg_pow_spec(At))
            
            # cluster
            if cluster:
                S1[i] = np.mean(S1t)
                S2[i] = np.mean(S2t)
            # END COMPUTE AVERAGES 
            
            # SUSTAINED ACTIVITY
            print('\nSimulating sustained activity...')
            self.r1 = 0.                        # suppress spontaneous excitation
            S = self.init_state(runs, 0.1) # initialize only one active node
            # TODO: nel paper dice che tutti gli altri sono inattivi, qui possono essere anche refrattari
            temp_tau = np.zeros((runs))
            
            # LOOP OVER TIME STEPS
            for t in tqdm(range(steps)):
                # update state vector
                S, s = self.update_state(S, W, T)
                temp_act = np.mean(s, axis=1)
                
                # if activity is suppressed, save timestep
                temp_tau += t*(temp_act==0)*(temp_tau==0)
                
                # if no activity in all runs, stop loop over time
                if np.count_nonzero(temp_tau==0) == 0:
                    break
            
            # if still activity, set tau equal to exp duration
            temp_tau += steps*(temp_tau==0)
            
            tau[i], sigma_tau[i] = np.mean(temp_tau), np.std(temp_tau)
            
            # reset r1 parameter
            self.r1 = r1_old
            
            # END SUSTAINED ACTIVITY
            
            # SUSCEPTIBILITY
            print('\nSimulating susceptibility...')
            window = 20
            thresh = 0.03
            temp_chi = np.zeros(runs)
            temp_act = np.zeros((runs, window))
            S = self.init_state(runs, 0.8) # initialize 80% active nodes
            
            # LOOP OVER TIME STEPS
            for t in tqdm(range(steps)):
                # update state vector
                S, s = self.update_state(S, W, T)
                
                if t<window:
                    temp_act[:,t] = np.mean(s, axis=1)
                else:
                    temp_act = np.roll(temp_act, -1)
                    temp_act[:,-1] = np.mean(s, axis=1)
                    
                    sigma_act = np.std(temp_act, axis=1)

                    # if activity is suppressed, save timestep
                    temp_chi += t*(sigma_act<thresh)*(temp_chi==0)
                
                    # if no activity in all runs, stop loop over time
                    if np.count_nonzero(temp_chi==0) == 0:
                        break
            
            # if still high std, set chi equal to exp duration
            temp_chi += steps*(temp_chi==0)
            
            chi[i], sigma_chi[i] = np.mean(temp_chi), np.std(temp_chi)
            # END SUSCEPTIBILITY
        
        # END LOOP OVER TEMPERATUREs
        clear_output(wait=True)
        print(self.title + '\n')
        print('End simulating activity')
        
        # RESHAPE CLUSTER PDF
        pdf = [np.array(list(i.items())).T for i in pdf]
        pdf = [i[:, i[0].argsort()] for i in pdf]

        # RETURN RESULTS
        if cluster:
            return (A, sigma_A, C, sigma_C, ent, sigma_ent, 
                    ev, sigma_ev, tau, sigma_tau, chi, sigma_chi, 
                    spectr, act, S1/self.N, S2/self.N, pdf)
        else:
            return (A, sigma_A, C, sigma_C, ent, sigma_ent, 
                    ev, sigma_ev, tau, sigma_tau, chi, sigma_chi, 
                    spectr, act)
    
    
    def init_state(self, runs, fract):
        '''
        Initialize the state of the system
        fract: fraction of initial acrive neurons
        '''
        from math import ceil
        
        n_act = ceil(fract * self.N)     # number of initial active neurons
        
        # create vector with n_act 1's, the rest half 0's and half -1's
        ss = np.zeros(self.N)
        ss[:n_act] = 1.
        ss[-(self.N-n_act)//2:] = -1.
        
        # return shuffled array
        return np.array([np.random.choice(ss, len(ss), replace=False) for _ in range(runs)])
    
    
    def update_state(self, S, W, T):
        '''
        Update state of the system according to HTC model
        '''
        
        probs = np.random.rand(S.shape[0], S.shape[1])   # generate probabilities
        s = (S==1).astype(int)                           # get active nodes
        pA = self.r1 + (1.-self.r1) * ( (W@s.T)>T )      # prob. to become active

        # update state vector
        newS = ( (S==0)*(probs<pA.T)                 # I->A
            + (S==1)*-1                              # A->R
            + (S==-1)*(probs>self.r2)*-1 )           # R->I (remain R with prob 1-r2)
        
        return (newS, (newS==1).astype(int) )
    
    
    def save(self):
        '''
        Save output
        '''
        
        filename = results_folder+self.name
        
        # Save weights matrix
        np.savetxt(filename + delimiter + 'matrix.txt', self.W)
        # Save activity
        np.savetxt(filename + delimiter + 'series.txt', np.vstack((self.act, self.act_norm)), fmt='%e')
        
        # Save power spectrum
        with open(filename + delimiter + 'spectrum.txt', 'w') as outfile:
            for x in self.spectr:
                np.savetxt(outfile, x)
                outfile.write('\n')
            for x in self.spectr_norm:
                np.savetxt(outfile, x)
                outfile.write('\n')
        
        if not self.cluster:
            np.savetxt(filename + delimiter + 'observables.txt',
                       (self.A, self.sigmaA, self.C, self.sigmaC, self.Ent, self.sigmaEnt,
                        self.Ev, self.sigmaEv, self.Tau, self.sigmaTau, self.Chi, self.sigmaChi,
                        self.A_norm, self.sigmaA_norm, self.C_norm, self.sigmaC_norm, self.Ent_norm, self.sigmaEnt_norm,
                        self.Ev_norm, self.sigmaEv_norm, self.Tau_norm, self.sigmaTau_norm, self.Chi_norm, self.sigmaChi_norm
                        ), fmt='%e')
            
        else:
            np.savetxt(filename + delimiter + 'observables.txt',
                       (self.A, self.sigmaA, self.C, self.sigmaC, self.Ent, self.sigmaEnt,
                        self.Ev, self.sigmaEv, self.Tau, self.sigmaTau, self.Chi, self.sigmaChi,
                        self.S1, self.S2,
                        self.A_norm, self.sigmaA_norm, self.C_norm, self.sigmaC_norm, self.Ent_norm, self.sigmaEnt_norm,
                        self.Ev_norm, self.sigmaEv_norm, self.Tau_norm, self.sigmaTau_norm, self.Chi_norm, self.sigmaChi_norm,
                        self.S1_norm, self.S2_norm), fmt='%e')
                        
            with open(filename + delimiter + 'pdf.txt', 'w') as outfile:
                for x in self.pdf:
                    np.savetxt(outfile, x)
                    outfile.write('\n')
                for x in self.pdf_norm:
                    np.savetxt(outfile, x)
                    outfile.write('\n')

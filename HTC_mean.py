from HTC import HTC
from HTC_utils import reshape_pdf, get_Tc, normalize

import numpy as np
from scipy.linalg import eigh as largest_eig
from collections import Counter
from pathlib import Path

def list_to_counter(lst):
    '''
    Convert an occurences list to a Counter
    '''
    # Recover values
    values = [[y]*int(x) for x, y in zip(lst[0], lst[1])]
    # Return Counter from flattened list.
    return Counter( [z for s in values for z in s] )


def get_mean_HTC(folder, name, N):
    '''
    Return mean HTC object from many realization of the same graph ensemble
    Clean the related files, leaving only '*_observables.txt'
    '''
    
    print('----------- GET HTC MEAN -----------')
    print(name, '\n')
    
    # Load first HTC model
    mean = HTC.loadFromName(folder+name+'_'+str(0))
    # Change Id
    mean.Id = -1
    mean.name = mean.name.rsplit('_', 1)[0] + '_' + str(mean.Id)
    
    # Initialize list for saving In-Degree, Lambda, Tc's
    #param = np.zeros(( len(mean.Trange), 3 ))
    #param_norm = np.zeros(( len(mean.Trange), 3 ))
    
    # Init list for saving Tc's
    Tc = np.zeros(N)
    Tc_norm = np.zeros(N)
        
    # Load pdfs
    pdf = [list_to_counter(x) for x in mean.pdf]
    pdf_norm = [list_to_counter(x) for x in mean.pdf_norm]
    
    '''
    pdf_ev = [list_to_counter(x) for x in mean.pdf_ev]
    pdf_ev_norm = [list_to_counter(x) for x in mean.pdf_ev_norm]
    pdf_size = [list_to_counter(x) for x in mean.pdf_size]
    pdf_size_norm = [list_to_counter(x) for x in mean.pdf_size_norm]
    pdf_time = [list_to_counter(x) for x in mean.pdf_time]
    pdf_time_norm = [list_to_counter(x) for x in mean.pdf_time_norm]
    pdf_size_causal = [list_to_counter(x) for x in mean.pdf_size_causal]
    pdf_size_causal_norm = [list_to_counter(x) for x in mean.pdf_size_causal_norm]
    pdf_time_causal = [list_to_counter(x) for x in mean.pdf_time_causal]
    pdf_time_causal_norm = [list_to_counter(x) for x in mean.pdf_time_causal_norm]
    '''

    for i in range(1,N):
        # Load model
        mod = HTC.loadFromName(folder+name+'_'+str(i))
        # Standard
        mean.A += mod.A
        mean.sigmaA += mod.sigmaA
        mean.Fisher += mod.Fisher
        mean.S1 += mod.S1
        mean.S2 += mod.S2
    
        # Normalized
        mean.A_norm += mod.A_norm
        mean.sigmaA_norm += mod.sigmaA_norm
        mean.Fisher_norm += mod.Fisher_norm
        mean.S1_norm += mod.S1_norm
        mean.S2_norm += mod.S2_norm
        
        # Cluster pdf
        pdf[i] += list_to_counter(mod.pdf[i])
        pdf_norm[i] += list_to_counter(mod.pdf_norm[i])
        
        # Tc
        #param[i, 0], param_norm[i, 0] = get_Tc(mod)
        Tc[i], Tc_norm[i] = get_Tc(mod)
        
        '''
        # Degree
        W, W_norm = mod.W, normalize(mod.W)
        
        g = igraph.Graph.Adjacency((mod.W > 0).tolist())
        g.es['weight'] = W[W.nonzero()]
        deg = np.mean(g.strength(mode='IN', weights = g.es['weight']))
        deg_norm = 1.
        param[i, 1], param_norm[i, 1] = deg, deg_norm
        
        # Lambda
        lmbd = largest_eig(W, eigvals=(W.shape[0]-1,W.shape[0]-1), eigvals_only=True)
        lmbd_norm = largest_eig(W_norm, eigvals=(W.shape[0]-1,W.shape[0]-1), eigvals_only=True)
        param[i, 2], param_norm[i, 2] = lmbd, lmbd_norm
        '''
        
        '''
        # Dinamical range
        mean.Exc += mod.Exc
        mean.Exc_norm += mod.Exc_norm
        
        for i in range(len(pdf_ev)):
            # Spectr
            mean.spectr[i][1] += mod.spectr[i][1]
            mean.spectr_norm[i][1] += mod.spectr_norm[i][1]

            # Pdfs
            pdf_ev[i] += list_to_counter(mod.pdf_ev[i])
            pdf_ev_norm[i] += list_to_counter(mod.pdf_ev_norm[i])
            pdf_size[i] += list_to_counter(mod.pdf_size[i])
            pdf_size_norm[i] += list_to_counter(mod.pdf_size_norm[i])
            pdf_time[i] += list_to_counter(mod.pdf_time[i])
            pdf_time_norm[i] += list_to_counter(mod.pdf_time_norm[i])
            pdf_size_causal[i] += list_to_counter(mod.pdf_size_causal[i])
            pdf_size_causal_norm[i] += list_to_counter(mod.pdf_size_causal_norm[i])
            pdf_time_causal[i] += list_to_counter(mod.pdf_time_causal[i])
            pdf_time_causal_norm[i] += list_to_counter(mod.pdf_time_causal_norm[i])
            
            # Pdf aval
         '''
    # Divide by N all the variables
    mean.A, mean.sigmaA, mean.Fisher, mean.S1, mean.S2 = \
    mean.A/N, mean.sigmaA/N, mean.Fisher/N, mean.S1/N, mean.S2/N

    # Divide by N all the normalized variables
    mean.A_norm, mean.sigmaA_norm, mean.Fisher_norm, mean.S1_norm, mean.S2_norm = \
    mean.A_norm/N, mean.sigmaA_norm/N, mean.Fisher_norm/N, mean.S1_norm/N, mean.S2_norm/N
    
    '''
    # Dinamical range
    mean.Exc /= N
    mean.Exc_norm /= N
    
    # Divide by N power spectrum
    for i in range(len(pdf_ev)):
        mean.spectr[i][1] /= N
        mean.spectr_norm[i][1] /= N
    '''
    
    # Store pdfs
    mean.pdf = reshape_pdf(pdf)
    mean.pdf_norm = reshape_pdf(pdf_norm)
    '''
    mean.pdf_ev = reshape_pdf(pdf_ev)
    mean.pdf_ev_norm = reshape_pdf(pdf_ev_norm)
    mean.pdf_size = reshape_pdf(pdf_size)
    mean.pdf_size_norm = reshape_pdf(pdf_size_norm)
    mean.pdf_time = reshape_pdf(pdf_time)
    mean.pdf_time_norm = reshape_pdf(pdf_time_norm)
    mean.pdf_size_causal = reshape_pdf(pdf_size_causal)
    mean.pdf_size_causal_norm = reshape_pdf(pdf_size_causal_norm)
    mean.pdf_time_causal = reshape_pdf(pdf_time_causal)
    mean.pdf_time_causal_norm = reshape_pdf(pdf_time_causal_norm)
    '''
    
    # Clean folder
    path = Path(folder)
    for file in path.iterdir():
        if name in str(file):
            if not 'observables' in str(file):
                file.unlink()
                
    # Save mean object            
    mean.save(folder, cluster=True, dinamical=False, complete_simulation=False)
    # Save Tc's
    np.savetxt(folder+name+'_'+'Tc.txt', np.vstack([Tc, Tc_norm]), fmt='%e')
    
    return mean, Tc, Tc_norm
    
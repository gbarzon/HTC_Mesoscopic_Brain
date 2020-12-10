from HTC import HTC
from HTC_utils import reshape_pdf
from collections import Counter
from pathlib import Path

def list_to_counter(lst):
    '''
    Convert an occurences list to a Counter
    '''
    # Recover values based on assumption 1.
    values = [[y]*int(x) for x, y in zip(lst[0], lst[1])]
    # Return Counter from flattened list.
    return Counter( [z for s in values for z in s] )


def get_mean_HTC(folder, name, N):
    '''
    Return mean HTC object from many realization of the same graph ensemble
    Clean the related files, leaving only '*_observables.txt'
    '''
    
    # Initialize list for saving TCs
    
    # Load first HTC model
    mean = HTC.loadFromName(folder+name+str(0))
    # Change Id
    mean.Id = -1
    mean.name = mean.name.rsplit('_', 1)[0] + '_' + str(mean.Id)
    
    # Load pdfs
    pdf_ev = [list_to_counter(x) for x in mean.pdf_ev]
    pdf_tau = [list_to_counter(x) for x in mean.pdf_tau]
    pdf_ev_norm = [list_to_counter(x) for x in mean.pdf_ev_norm]
    pdf_tau_norm = [list_to_counter(x) for x in mean.pdf_tau_norm]
    pdf = [list_to_counter(x) for x in mean.pdf]
    pdf_norm = [list_to_counter(x) for x in mean.pdf_norm]

    for i in range(1,N):
        # Load model
        mod = HTC.loadFromName(folder+name+str(i))
        # Standard
        mean.A += mod.A
        mean.sigmaA += mod.sigmaA
        mean.C += mod.C
        mean.sigmaC += mod.sigmaC
        mean.Ent += mod.Ent
        mean.sigmaEnt += mod.sigmaEnt
        mean.Ev += mod.Ev
        mean.sigmaEv += mod.sigmaEv
        mean.Tau += mod.Tau
        mean.sigmaTau += mod.sigmaTau
        mean.Chi += mod.Chi
        mean.sigmaChi += mod.sigmaChi
        mean.S1 += mod.S1
        mean.S2 += mod.S2
    
        # Normalized
        mean.A_norm += mod.A_norm
        mean.sigmaA_norm += mod.sigmaA_norm
        mean.C_norm += mod.C_norm
        mean.sigmaC_norm += mod.sigmaC_norm
        mean.Ent_norm += mod.Ent_norm
        mean.sigmaEnt_norm += mod.sigmaEnt_norm
        mean.Ev_norm += mod.Ev_norm
        mean.sigmaEv_norm += mod.sigmaEv_norm
        mean.Tau_norm += mod.Tau_norm
        mean.sigmaTau_norm += mod.sigmaTau_norm
        mean.Chi_norm += mod.Chi_norm
        mean.sigmaChi_norm += mod.sigmaChi_norm
        mean.S1_norm += mod.S1_norm
        mean.S2_norm += mod.S2_norm
        
        for i in range(len(pdf_ev)):
            # Spectr
            mean.spectr[i][1] += mod.spectr[i][1]
            mean.spectr_norm[i][1] += mod.spectr_norm[i][1]

            # Pdfs
            pdf_ev[i] += list_to_counter(mod.pdf_ev[i])
            pdf_tau[i] += list_to_counter(mod.pdf_tau[i])
            pdf_ev_norm[i] += list_to_counter(mod.pdf_ev_norm[i])
            pdf_tau_norm[i] += list_to_counter(mod.pdf_tau_norm[i])
            pdf[i] += list_to_counter(mod.pdf[i])
            pdf_norm[i] += list_to_counter(mod.pdf_norm[i])
    
    # Divide by N all the variables
    mean.A, mean.sigmaA, mean.C, mean.sigmaC, \
    mean.Ent, mean.sigmaEnt, mean.Ev, mean.sigmaEv, \
    mean.Tau, mean.sigmaTau, mean.Chi, mean.sigmaChi, \
    mean.S1, mean.S2 = \
    mean.A/N, mean.sigmaA/N, mean.C/N, mean.sigmaC/N, \
    mean.Ent/N, mean.sigmaEnt/N, mean.Ev/N, mean.sigmaEv/N, \
    mean.Tau/N, mean.sigmaTau/N, mean.Chi/N, mean.sigmaChi/N, \
    mean.S1/N, mean.S2/N

    # Divide by N all the normalized variables
    mean.A_norm, mean.sigmaA_norm, mean.C_norm, mean.sigmaC_norm, \
    mean.Ent_norm, mean.sigmaEnt_norm, mean.Ev_norm, mean.sigmaEv_norm, \
    mean.Tau_norm, mean.sigmaTau_norm, mean.Chi_norm, mean.sigmaChi_norm, \
    mean.S1_norm, mean.S2_norm = \
    mean.A_norm/N, mean.sigmaA_norm/N, mean.C_norm/N, mean.sigmaC_norm/N, \
    mean.Ent_norm/N, mean.sigmaEnt_norm/N, mean.Ev_norm/N, mean.sigmaEv_norm/N, \
    mean.Tau_norm/N, mean.sigmaTau_norm/N, mean.Chi_norm/N, mean.sigmaChi_norm/N, \
    mean.S1_norm/N, mean.S2_norm/N
    
    # Divide by N power spectrum
    for i in range(len(pdf_ev)):
        mean.spectr[i][1] /= N
        mean.spectr_norm[i][1] /= N

    # Store pdfs
    mean.pdf_ev = reshape_pdf(pdf_ev)
    mean.pdf_tau = reshape_pdf(pdf_tau)
    mean.pdf_ev_norm = reshape_pdf(pdf_ev_norm)
    mean.pdf_tau_norm = reshape_pdf(pdf_tau_norm)
    mean.pdf = reshape_pdf(pdf)
    mean.pdf_norm = reshape_pdf(pdf_norm)
    
    
    # Clean folder
    path = Path(folder)
    for file in path.iterdir():
        if name in str(file):
            if not 'observables' in str(file):
                file.unlink()
                
    mean.save(folder, cluster=True, dinamical=False)
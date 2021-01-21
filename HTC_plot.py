import numpy as np
import networkx as nx
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt

from matplotlib import cm
from HTC_utils import find_nearest

class plotHTC:
    
    def __init__(self):
        print('Init plotter')
        #self.model = model
        
        
    def template_plot(self, Trange, Tc, Tc_norm, var1, var1_norm, name1,
                      var2=None, var2_norm=None, name2=None,
                      title=None, derivative=False):
        fig, ax1 = plt.subplots(figsize=(12,6))
        plt.grid()

        ax1.plot(Trange/Tc, var1, label=r'${}$'.format(name1), c='black')
        ax1.plot(Trange/Tc_norm, var1_norm, label=r'${}_{{norm}}$'.format(name1), c='red')
        ax1.set_xlabel(r'$T/T_c$', size=12)
        ax1.set_ylabel(r'${}$'.format(name1), size=12)

        ax2 = ax1.twinx()
        if not var2 is None:
            if derivative:
                ax2.plot(Trange[1:]/Tc, var2, '-.', label=r'${}$'.format(name2), c='black')
                ax2.plot(Trange[1:]/Tc_norm, var2_norm, '-.', label=r'${}_{{norm}}$'.format(name2), c='red')
            else:
                ax2.plot(Trange/Tc, var2, '-.', label=r'${}$'.format(name2), c='black')
                ax2.plot(Trange/Tc_norm, var2_norm, '-.', label=r'${}_{{norm}}$'.format(name2), c='red')
            ax2.set_ylabel(r'${}$'.format(name2), size=12)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2, fontsize=12)
        plt.title(r'${}$'.format(title), size=13)

        fig.tight_layout()
        plt.show()
        
        
    def plot_variable(self, model, var_to_print):
        Tc = np.argmax(model.S2) / (len(model.Trange)-1) * model.Tmax
        Tc_norm = np.argmax(model.S2_norm) / (len(model.Trange)-1) * model.Tmax
        
        if var_to_print == 'act':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.A, model.A_norm, 'A',
                              model.sigmaA, model.sigmaA_norm, '\sigma(A)', model.title)
        elif var_to_print == 'cluster':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.S1, model.S1_norm, 'S1',
                              model.S2, model.S2_norm, 'S2', model.title)
        elif var_to_print == 'fisher':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.Fisher, model.Fisher_norm, 'Fisher \  information',
                               title=model.title)
        elif var_to_print == 'ent':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.Ent, model.Ent_norm, 'Ent',
                              model.sigmaEnt, model.sigmaEnt_norm, '\sigma(Ent)', model.title)
        elif var_to_print == 'interevent':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.Ev, model.Ev_norm, 't(ev)',
                              (model.Ev[1:]-model.Ev[:-1])/model.dT,
                               (model.Ev_norm[1:]-model.Ev_norm[:-1])/model.dT,
                               't\prime(ev)', model.title, derivative=True)
        elif var_to_print == 'tau':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.Tau, model.Tau_norm, '\\tau',
                               -(model.Tau[1:]-model.Tau[:-1])/model.dT,
                               -(model.Tau_norm[1:]-model.Tau_norm[:-1])/model.dT,
                               '- \\tau \prime', model.title, derivative=True)
        elif var_to_print == 'chi':
            self.template_plot(model.Trange, model.Tc*Tc, model.Tc*Tc_norm, model.Chi, model.Chi_norm, '\\chi',
                               title=model.title)
            
        
    def plot_series(self, model):
        import matplotlib.ticker as ticker

        plt.figure(figsize=(16,16))
        
        # length of sketch of time series to be plotted
        N = 200
        
        Ts = [len(model.Trange)//5, np.argmax(model.S2_norm), len(model.Trange)-len(model.Trange)//4]
        
        for i in range(len(Ts)):
            # plot time series
            ax = plt.subplot(4, len(Ts), i+1)
            
            plt.title('T='+str(round(Ts[i]/np.argmax(model.S2_norm),2))+'*Tc', size=13)
            plt.plot(range(N), model.act[Ts[i]][100:N+100], c='black', lw=0.8)
            plt.xlabel('t', size=13)
            plt.ylim( [-0.05, 0.42] )
            plt.grid()
            
            if i==len(Ts)-1:
                ax.yaxis.tick_right()
            else:
                ax.set_yticklabels([])
            if i==0:
                plt.ylabel(r'$<A(t)>$', size=13)
                
            # plot normalized time series
            ax = plt.subplot(4, len(Ts), i+1+len(Ts))
            
            plt.plot(range(N), model.act_norm[Ts[i]][100:N+100], c='red', lw=0.8)
            plt.xlabel('t', size=13)
            plt.ylim( [-0.05, 0.42] )
            plt.grid()
            
            if i==len(Ts)-1:
                ax.yaxis.tick_right()
            else:
                ax.set_yticklabels([])
            if i==0:
                plt.ylabel(r'$<A_{norm}(t)>$', size=13)
            
            # plot power spectrum
            ax = plt.subplot(4, len(Ts), i+1+2*len(Ts))
            
            freq = np.arange(len(model.spectr[0])) / len(model.spectr[0])
            
            plt.plot(freq, model.spectr[Ts[i]], alpha=0.7, label=r'$P$', c='black')
            plt.plot(freq, model.spectr_norm[Ts[i]], alpha=0.7, label=r'$P_{norm}$', c='red')
            plt.xlabel('f', size=13)
            
            plt.ylim( [0.0, np.max(model.spectr_norm[Ts[1]]) + 0.1*np.max(model.spectr_norm[Ts[1]])] )
            plt.grid()
            
            if i==len(Ts)-1:
                plt.legend()
                y_labels = ax.get_yticks()
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
                ax.yaxis.tick_right()
            else:
                ax.set_yticklabels([])
            if i==0:
                plt.ylabel(r'$P(f)$', size=13)
                
            # plot autocorrelation
            nlags = 30
            ax = plt.subplot(4, len(Ts), i+1+3*len(Ts))
            
            plt.plot(acf(model.act[Ts[i]], nlags=nlags, fft=False), alpha=0.7, label=r'$Acf(\tau)$', c='black')
            plt.plot(acf(model.act_norm[Ts[i]], nlags=nlags, fft=False),
                     alpha=0.7, label=r'$Acf_{norm}(\tau)$', c='red')
            
            plt.grid()
            
            if i==len(Ts)-1:
                plt.legend()
                y_labels = ax.get_yticks()
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
                ax.yaxis.tick_right()
            else:
                ax.set_yticklabels([])
            if i==0:
                plt.ylabel(r'$Acf(\tau)$', size=13)
              
            '''
            # plot cluster distribution
            ax = plt.subplot(4, len(Ts), i+1+3*len(Ts))
            
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.scatter(model.pdf[Ts[i]][0], model.pdf[Ts[i]][1]/np.sum(model.pdf[Ts[i]][1]), alpha=0.7, label=r'$pdf(s)$', c='black')
            plt.scatter(model.pdf_norm[Ts[i]][0], model.pdf_norm[Ts[i]][1]/np.sum(model.pdf_norm[Ts[i]][1]),
                     alpha=0.7, label=r'$pdf_{norm}(s)$', c='red')
            
            plt.grid()
            
            if i==len(Ts)-1:
                plt.legend()
                y_labels = ax.get_yticks()
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
                ax.yaxis.tick_right()
            else:
                ax.set_yticklabels([])
            if i==0:
                plt.ylabel(r'$pdf(s)$', size=13)
            '''
        plt.show()
    
    def plot_pdf(self, mod, xlabel, Nbins=None, yrange=None, scale='loglog'):
        cm1 = cm.get_cmap('jet')
        fact = 8
        
        Trange = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2) * (len(mod.Trange)-1)
        Trange_norm = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2_norm) * (len(mod.Trange)-1)
        
        if xlabel == 'cluster':
            pdf, pdf_norm = mod.pdf, mod.pdf_norm
            xlabel = 'S'
        elif xlabel == 'ev':
            pdf, pdf_norm = mod.pdf_ev, mod.pdf_ev_norm
            xlabel = r'$t_{ev}$'
        elif xlabel == 'size':
            pdf, pdf_norm = mod.pdf_size, mod.pdf_size_norm
            xlabel = r'$size$'
        elif xlabel == 'size':
            pdf, pdf_norm = mod.pdf_size, mod.pdf_size_norm
            xlabel = r'$size$'
        elif xlabel == 'time':
            pdf, pdf_norm = mod.pdf_time, mod.pdf_time_norm
            xlabel = r'$time$'
        elif xlabel == 'size_causal':
            pdf, pdf_norm = mod.pdf_size_causal, mod.pdf_size_causal_norm
            xlabel = r'$size, causal$'
        elif xlabel == 'time_causal':
            pdf, pdf_norm = mod.pdf_time_causal, mod.pdf_time_causal_norm
            xlabel = r'$time, causal$'
        
        plt.figure(figsize=(20,6))
        
        # Pdf
        plt.subplot(1, 2, 1)
        
        if scale == 'log':
            plt.yscale('log')
        elif scale == 'loglog':
            plt.xscale('log')
            plt.yscale('log')

        for i in range(2,8):
            val = 0.2 * i
            ind = find_nearest(Trange, val)
            
            if Nbins==None:
                plt.scatter(pdf[ind][0], pdf[ind][1]/np.sum(pdf[ind][1]),
                            label='T='+str(round(val,1))+'*Tc', alpha=0.8, s=20,
                            c=[cm1(i/fact)]*len(pdf[ind][0]))
            else:
                ist = np.histogram(pdf[ind][0], weights=pdf[ind][1]/np.sum(pdf[ind][1]), bins=Nbins)
                plt.scatter( (ist[1][1:] + ist[1][:-1])/2, ist[0], label='T='+str(round(val,1))+'*Tc', s=20,
                            c=[cm1(i/fact)]*len(ist[0]))
                
        plt.xlabel(xlabel, fontsize=13)
        plt.ylabel('pdf', fontsize=13)
        if not yrange is None:
            plt.ylim(yrange)
        plt.grid()
        plt.legend(title='W', fontsize=13)
        
        # Pdf norm
        plt.subplot(1, 2, 2)
        
        if scale == 'log':
            plt.yscale('log')
        elif scale == 'loglog':
            plt.xscale('log')
            plt.yscale('log')

        for i in range(2,8):
            val = 0.2 * i
            ind = find_nearest(Trange, val)
        
            if Nbins==None:
                plt.scatter(pdf_norm[ind][0], pdf_norm[ind][1]/np.sum(pdf_norm[ind][1]),
                            label='T='+str(round(val,1))+'*Tc',alpha=0.8, s=20,
                            c=[cm1(i/fact)]*len(pdf_norm[ind][0]))
            else:
                ist = np.histogram(pdf_norm[ind][0], weights=pdf_norm[ind][1]/np.sum(pdf_norm[ind][1]), bins=Nbins)
                plt.scatter( (ist[1][1:] + ist[1][:-1])/2, ist[0], label='T='+str(round(val,1))+'*Tc', s=20,
                            c=[cm1(i/fact)]*len(ist[0]))
        
        plt.xlabel(xlabel, fontsize=13)
        plt.ylabel('pdf', fontsize=13)
        if not yrange is None:
            plt.ylim(yrange)
        plt.grid()
        plt.legend(title=r'$W_{norm}$', fontsize=13)
        plt.show()
        
    def plot_stimulated(self, mod):
        plt.figure(figsize=(20,6))

        Trange = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2) * (len(mod.Trange)-1)
        Trange_norm = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2_norm) * (len(mod.Trange)-1)

        #cm1 = cm.get_cmap('viridis')
        #cm1 = cm.get_cmap('gray')
        #cm1 = cm.get_cmap('autumn')
        cm1 = cm.get_cmap('jet')
        fact = 8

        # Act vs. stimuli - original network
        plt.subplot(1, 2, 1)
        plt.xscale('log')
        #plt.yscale('log')
        plt.grid()
        plt.xlabel('stimulus s', size=13)
        plt.ylabel('A', size=13)

        for i in range(2,8):
            val = 0.2 * i
            ind = find_nearest(Trange, val)
    
            plt.plot(mod.stimuli, mod.Exc[ind], lw=2, c=cm1(i/fact), label='T='+str(round(val,1))+'*Tc')
        plt.legend(fontsize=12, title=r'W')
        
        # Act vs. stimuli - normalized network
        plt.subplot(1, 2, 2)
        plt.xscale('log')
        #plt.yscale('log')
        plt.grid()
        plt.xlabel('stimulus s', size=13)
        plt.ylabel('A', size=13)

        for i in range(2,8):
            val = 0.2 * i
            ind = find_nearest(Trange_norm, val)
    
            plt.plot(mod.stimuli, mod.Exc_norm[ind], '-.', lw=2, c=cm1(i/fact), label='T='+str(round(val,1))+'*Tc')
        plt.legend(fontsize=12, title=r'$W_{norm}$')
    
    
    def plot_dynamical_range(self, mod, low=0.15, high=0.85):
        plt.figure(figsize=(20,6))

        Trange = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2) * (len(mod.Trange)-1)
        Trange_norm = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2_norm) * (len(mod.Trange)-1)

        #cm1 = cm.get_cmap('viridis')
        #cm1 = cm.get_cmap('gray')
        #cm1 = cm.get_cmap('autumn')
        cm1 = cm.get_cmap('jet')
        fact = 8

        # Act vs. stimuli - original network
        plt.subplot(1, 3, 1)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.xlabel('stimulus s', size=13)
        plt.ylabel('A', size=13)

        for i in range(2,8):
            val = 0.2 * i
            ind = find_nearest(Trange, val)
    
            plt.plot(mod.stimuli, mod.Exc[ind], lw=2, c=cm1(i/fact), label='T='+str(round(val,1))+'*Tc')
        plt.legend(fontsize=12, title=r'W')
        
        # Act vs. stimuli - normalized network
        plt.subplot(1, 3, 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.xlabel('stimulus s', size=13)
        plt.ylabel('A', size=13)

        for i in range(2,8):
            val = 0.2 * i
            ind = find_nearest(Trange_norm, val)
    
            plt.plot(mod.stimuli, mod.Exc_norm[ind], '-.', lw=2, c=cm1(i/fact), label='T='+str(round(val,1))+'*Tc')
        plt.legend(fontsize=12, title=r'$W_{norm}$')
    
        # Dynamical range
        from HTC_utils import get_dynamical_range
        delta, delta_norm = get_dynamical_range(mod, low, high)

        
        Trange = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2) * (len(mod.Trange)-1)
        Trange_norm = mod.Trange / mod.Tc / mod.Tmax / np.argmax(mod.S2_norm) * (len(mod.Trange)-1)
        plt.subplot(1, 3, 3)
        plt.scatter(Trange, delta, c='k', s=18, alpha=0.6, label=r'$W$')
        #plt.plot(mod.Trange/mod.Tc, delta, c='k')
        plt.scatter(Trange_norm, delta_norm, c='r', s=18, alpha=0.5, label=r'$W_{norm}$')
        #plt.plot(mod.Trange/mod.Tc, delta_norm, c='r')

        plt.xlabel(r'$T/T_c$', size=13)
        plt.ylabel(r'$ dynamical \ range \ \Delta$', size=13)
        plt.grid()
        plt.legend(fontsize=12)

        plt.show()
    
    
    def draw_network(self, model):
        G = nx.from_numpy_matrix(model.W)
        pos = nx.spring_layout(G)  # positions for all nodes
        
        fig, _ = plt.subplots(figsize=(12,6))

        nx.draw_networkx_nodes(G, pos, node_color="r", node_size=300)
        for edge in G.edges(data='weight'):
            #nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2]*20)
            nx.draw_networkx_edges(G, pos, edgelist=[edge])
            
        plt.axis('off')
        plt.show()

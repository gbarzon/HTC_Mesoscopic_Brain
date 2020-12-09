import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

class plotHTC:
    
    def __init__(self):
        print('Init plotter')
        #self.model = model
        
        
    def template_plot(self, Trange, Tc, var1, var1_norm, name1,
                      var2=None, var2_norm=None, name2=None,
                      title=None, derivative=False):
        fig, ax1 = plt.subplots(figsize=(12,6))
        plt.grid()

        ax1.plot(Trange/Tc, var1, label=r'${}$'.format(name1), c='black')
        ax1.plot(Trange/Tc, var1_norm, label=r'${}_{{norm}}$'.format(name1), c='red')
        ax1.set_xlabel(r'$T/T_c$', size=12)
        ax1.set_ylabel(r'${}$'.format(name1), size=12)

        ax2 = ax1.twinx()
        if not var2 is None:
            if derivative:
                ax2.plot(Trange[1:]/Tc, var2, '-.', label=r'${}$'.format(name2), c='black')
                ax2.plot(Trange[1:]/Tc, var2_norm, '-.', label=r'${}_{{norm}}$'.format(name2), c='red')
            else:
                ax2.plot(Trange/Tc, var2, '-.', label=r'${}$'.format(name2), c='black')
                ax2.plot(Trange/Tc, var2_norm, '-.', label=r'${}_{{norm}}$'.format(name2), c='red')
            ax2.set_ylabel(r'${}$'.format(name2), size=12)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2, fontsize=12)
        plt.title(r'${}$'.format(title), size=13)

        fig.tight_layout()
        plt.show() 
        
        
    def plot_variable(self, model, var_to_print):
        if var_to_print == 'act':
            self.template_plot(model.Trange, model.Tc, model.A, model.A_norm, 'A',
                              model.sigmaA, model.sigmaA_norm, '\sigma(A)', model.title)
        elif var_to_print == 'cluster':
            self.template_plot(model.Trange, model.Tc, model.S1, model.S1_norm, 'S1',
                              model.S2, model.S2_norm, 'S2', model.title)
        elif var_to_print == 'corr':
            self.template_plot(model.Trange, model.Tc, model.C, model.C_norm, 'C',
                              model.sigmaC, model.sigmaC_norm, '\sigma(C)', model.title)
        elif var_to_print == 'ent':
            self.template_plot(model.Trange, model.Tc, model.Ent, model.Ent_norm, 'Ent',
                              model.sigmaEnt, model.sigmaEnt_norm, '\sigma(Ent)', model.title)
        elif var_to_print == 'interevent':
            self.template_plot(model.Trange, model.Tc, model.Ev, model.Ev_norm, 't(ev)',
                              (model.Ev[1:]-model.Ev[:-1])/model.dT,
                               (model.Ev_norm[1:]-model.Ev_norm[:-1])/model.dT,
                               't\prime(ev)', model.title, derivative=True)
        elif var_to_print == 'tau':
            self.template_plot(model.Trange, model.Tc, model.Tau, model.Tau_norm, '\\tau',
                               -(model.Tau[1:]-model.Tau[:-1])/model.dT,
                               -(model.Tau_norm[1:]-model.Tau_norm[:-1])/model.dT,
                               '- \\tau \prime', model.title, derivative=True)
        elif var_to_print == 'chi':
            self.template_plot(model.Trange, model.Tc, model.Chi, model.Chi_norm, '\\chi',
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
            
            plt.plot(model.spectr[Ts[i]][0], model.spectr[Ts[i]][1], alpha=0.7, label=r'$P$', c='black')
            plt.plot(model.spectr_norm[Ts[i]][0], model.spectr_norm[Ts[i]][1], alpha=0.7, label=r'$P_{norm}$', c='red')
            plt.xlabel('f', size=13)
            
            plt.ylim( [0.0, np.max(model.spectr_norm[Ts[1]][1]) + 0.1*np.max(model.spectr_norm[Ts[1]][1])] )
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from solar_rs import element2z_dict, z2element_dict, \
    logeps_sols, logeps_solr, logeps_sols_dict, logeps_solr_dict
import socket
from pathlib import Path
hostname = socket.gethostname()
relative_datadir = Path('Data/rproc')
if hostname == 'jerome-linux':
    hostdir = Path('/home/jerome/Documents/GitHub')
    datadir = hostdir/relative_datadir
elif hostname == 'sage2020':
    hostdir = Path('/home/jiangrz/hdd23')
    datadir = hostdir/relative_datadir

class AbundPlot():
    def __init__(self, datafile='logeps.csv'):
        self.load_file(datafile)
    
    def load_file(self, datafile):
        datapath = Path(datafile)
        if datapath.exists():
            self.logeps_path = datapath
        else:
            self.logeps_path = datadir/datafile
        self.df = pd.read_csv(self.logeps_path)
        self.objnames = self.df.loc[:, 'objname'].values

    def load_star(self, star_name='J1158+0734'):
        colnames = self.df.columns
        for _idx, _stardata in self.df.iterrows():
            if _stardata.objname.startswith(star_name):
                star_data = _stardata.values
                self.star_name = star_data[0]
                break
        if not hasattr(self, 'star_name'):
            print('This star(%s) is not included in this file! '%star_name)
            print('Here are all available stars:')
            print(self.objnames)
            return
        
        col_logeps = [_ in list(element2z_dict.keys()) for _ in colnames]
        col_elogeps = [(_.startswith('e')) and (_ != 'eFeH') for _ in colnames]
        star_logeps = star_data[col_logeps]
        flag_detect = ~pd.isna(star_logeps)
        star_logeps = star_data[col_logeps][flag_detect]
        star_elogeps = star_data[col_elogeps][flag_detect]
        star_el = self.df.columns.values[col_logeps][flag_detect]

        star_Z = np.array([element2z_dict[_] for _ in star_el])
        star_logeps_dict = {
            z2element_dict[_z]: 
            star_logeps[_idx] for _idx, _z in enumerate(star_Z)
        }
        star_elogeps_dict = {
            z2element_dict[_z]: 
            star_elogeps[_idx] for _idx, _z in enumerate(star_Z)
        }

        # solar_s scaled to Ba, solar_r scaled to Eu
        logeps_sols_scaled = logeps_sols - (
            logeps_sols_dict['Ba'] - star_logeps_dict['Ba'])
        logeps_solr_scaled = logeps_solr - (
            logeps_solr_dict['Eu'] - star_logeps_dict['Eu'])
        # Scale Matrix
        S_mat_00 = logeps_solr_dict['Eu'] - star_logeps_dict['Eu']
        S_mat_01 = logeps_sols_dict['Eu'] - star_logeps_dict['Eu']
        S_mat_10 = logeps_solr_dict['Ba'] - star_logeps_dict['Ba']
        S_mat_11 = logeps_sols_dict['Ba'] - star_logeps_dict['Ba']
        S_mat = np.array([
            [S_mat_00, S_mat_01], 
            [S_mat_10, S_mat_11]
        ])
        dS_mat = S_mat * np.log(10) * (-np.array([
            [star_elogeps_dict['Eu'], star_elogeps_dict['Eu']], 
            [star_elogeps_dict['Ba'], star_elogeps_dict['Ba']]
        ]))
        S_mat = np.power(10, S_mat)
        inv_S_mat = np.linalg.inv(S_mat)
        vec_k = np.dot(inv_S_mat, np.array([1, 1]).reshape(-1, 1))
        kr, ks = vec_k.reshape(-1)
        dvec_k = - np.dot(np.matmul(inv_S_mat, dS_mat), vec_k)
        dkr, dks = dvec_k.reshape(-1)
        # if kr < 0:
        #     kr, ks = 0, np.power(10, -S_mat_11)
        # elif ks < 0:
        #     kr, ks = 0, np.power(10, -S_mat_00)
        Nsolr = kr*np.power(10, logeps_solr)
        Nsolr[np.isnan(Nsolr)] = 0
        Nsols = ks*np.power(10, logeps_sols)
        Nsols[np.isnan(Nsols)] = 0
        logeps_solrs_scaled = np.log10(Nsolr+Nsols)

        plot_Z = [element2z_dict[_] for _ in logeps_solr_dict.keys()]
        flag_heavy = star_Z > 30
        star_logeps_heavy = star_logeps[flag_heavy]
        star_elogeps_heavy = star_elogeps[flag_heavy]
        star_Z_heavy = star_Z[flag_heavy]
        flag_star = np.in1d(plot_Z, star_Z)
        solr_detect = logeps_solr_scaled[flag_star]
        sols_detect = logeps_sols_scaled[flag_star]
        solrs_detect = logeps_solrs_scaled[flag_star]
        relative_residual = self.func_relative_residual(
            star_logeps_heavy, 
            solrs_detect, 
            star_elogeps_heavy)
        
        self.star_logeps_dict = star_logeps_dict
        self.star_elogeps_dict = star_elogeps_dict
        self.plot_Z = plot_Z
        self.logeps_solr_scaled = logeps_solr_scaled
        self.logeps_sols_scaled = logeps_sols_scaled
        self.logeps_solrs_scaled = logeps_solrs_scaled
        self.star_Z_heavy = star_Z_heavy
        self.star_logeps_heavy = star_logeps_heavy
        self.star_elogeps_heavy = star_elogeps_heavy
        self.solr_detect = solr_detect
        self.sols_detect = sols_detect
        self.kr, self.ks = kr, ks
        self.dkr, self.dks = dkr, dks
        self.solrs_detect = solrs_detect
        self.relative_residual = relative_residual

    def plot(self, star_name=None, fname=None, **kw):
        if star_name is None:
            # initialize
            if not hasattr(self, 'star_name'):
                self.load_star()
        else:
            if hasattr(self, 'star_name') and (star_name == self.star_name):
                pass
            else:
                # update star information
                self.load_star(star_name)
        
        plot_Z = self.plot_Z
        logeps_solr_scaled = self.logeps_solr_scaled
        logeps_sols_scaled = self.logeps_sols_scaled
        logeps_solrs_scaled = self.logeps_solrs_scaled
        star_Z_heavy = self.star_Z_heavy
        star_logeps_heavy = self.star_logeps_heavy
        star_elogeps_heavy = self.star_elogeps_heavy
        solr_detect = self.solr_detect
        sols_detect = self.sols_detect
        solrs_detect = self.solrs_detect
        relative_residual = self.relative_residual

        fig, ax = plt.subplots(
            1, 1, figsize=(15, 7), 
            # height_ratios=(2, .9), 
            dpi=100, sharex=True)
        fig.subplots_adjust(hspace=.1)

        ax.plot(
            plot_Z, logeps_solr_scaled, 
            linewidth=3, c='salmon', alpha=.5, 
            label="Solar r-process(scaled to Eu)", 
            zorder=2)
        ax.plot(
            plot_Z, logeps_sols_scaled, 
            linewidth=3, c='dodgerblue', alpha=.5, 
            label="Solar s-process(scaled to Ba)", 
            zorder=2)
        scaled_kr = 100*self.kr/(self.kr+self.ks)
        scaled_ks = 100*self.ks/(self.kr+self.ks)
        ax.plot(plot_Z, logeps_solrs_scaled, 
            linewidth=3, c='k', #linestyle='--',
            label="Solar r(%.1f%%)+s(%.1f%%)"%(scaled_kr, scaled_ks))
        ec = 'darkgoldenrod'
        ew = 2
        ax.errorbar(
            star_Z_heavy, star_logeps_heavy, yerr=star_elogeps_heavy,
            ecolor=ec, elinewidth=ew, linestyle='', capsize=5, capthick=3,
            marker='s', markersize=10, mfc='orange', mec=ec, mew=ew,
            label='Observed value', 
            alpha=1, zorder=4)
        for _idx, _x in enumerate(star_Z_heavy):
            # _y = star_logeps_heavy[_idx] + .3
            # _t = z2element_dict[_x]
            # cover_threshold = .1
            # flag_cover = (
            #     np.abs(_y - solr_detect[_idx]) < cover_threshold) or (
            #     np.abs(_y - sols_detect[_idx]) < cover_threshold) or (
            #     np.abs(_y - solrs_detect[_idx]) < cover_threshold)
            # if  flag_cover:
            #     _y = star_logeps_heavy[_idx] - .3
            dist = .3
            _t = z2element_dict[_x]
            idx_x = np.argmin(np.abs(plot_Z - _x))
            _y = logeps_solrs_scaled[idx_x]
            sign_pos = +1
            if (idx_x != 0) and (idx_x != len(plot_Z)-1):
                if (logeps_solrs_scaled[idx_x-1] > _y) and (logeps_solrs_scaled[idx_x+1] > _y):
                    sign_pos = -1
            if sign_pos < 0:
                _y = np.min([
                    star_logeps_heavy[_idx], 
                    logeps_solr_scaled[idx_x], 
                    _y])
            else:
                _y = np.max([
                    star_logeps_heavy[_idx], 
                    logeps_solr_scaled[idx_x], 
                    _y])
            _y += (sign_pos * dist)
            ax.text(_x, _y, _t, ha='center', va='center', fontsize=14, zorder=5)
        ax.grid(True, linestyle='--', alpha=1, zorder=1)

        # axes[1].scatter(
        #     star_Z_heavy, np.abs(relative_residual), 
        #     marker='s', s=100, color='orange', edgecolor='k', linewidth=1, zorder=4)
        # axes[1].hlines(
        #     xmin=30, xmax=91, y=[1, 3], 
        #     colors='grey', linestyles=(0, (10, 5)), alpha=1, zorder=1)
        # for _idx, _x in enumerate(star_Z_heavy):
        #     _y = np.abs(relative_residual)[_idx] + .7
        #     _t = z2element_dict[_x]
        #     axes[1].text(_x, _y, _t, ha='center', va='center', fontsize=14, zorder=5)

        x_major_locator = MultipleLocator(5)
        y0_major_locator = MultipleLocator(1)
        # y1_major_locator = MultipleLocator(5)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y0_major_locator)
        # axes[1].set_yticks([0, 1, 3, 5])
        # axes[1].yaxis.set_major_locator(y1_major_locator)
        ax.set_xlim(31, 91)
        # axes[1].set_yscale('log')
        ax.tick_params(axis='both', labelsize=18)
        # axes[1].tick_params(axis='both', labelsize=18)

        ax.set_xlabel("Atomic Number(Z)", size=30)
        ax.set_ylabel(r'$\log\varepsilon$', size=30)
        # axes[1].set_ylabel(r'$\Delta$ log $\varepsilon$', fontdict={'size': 25})
        # axes[1].set_ylabel(r'$|\Delta\log\varepsilon|/\sigma$', fontdict={'size': 30})
        ax.legend(loc='best', fontsize=15, frameon=False)
        fig.suptitle(self.star_name, y=.95, size=30)
        if fname is not None:
            fig.savefig(fname, **kw)
        plt.close()
        return fig

    def redchisqr(self, star_name=None):
        if star_name is None:
            # initialize
            if not hasattr(self, 'star_name'):
                self.load_star()
            if hasattr(self, 'reduced_chisquare'):
                return self.reduced_chisquare
        else:
            if hasattr(self, 'star_name') and (star_name == self.star_name):
                pass
            else:
                # update star information
                self.load_star(star_name)
        if hasattr(self, 'relative_residual'):
            relative_residual = self.relative_residual
            reduced_chisquare = np.sum(np.square(relative_residual))
            star_Z_heavy = self.star_Z_heavy
            reduced_chisquare /= (len(star_Z_heavy) - 1)
            self.reduced_chisquare = reduced_chisquare
        return self.reduced_chisquare
    
    @staticmethod
    def func_relative_residual(obs, predict, err):
        return (obs-predict)/err

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import scipy.constants as const


class Dispersion_Analysis():

    def __init__(self, **kwargs):
        self.disp_file = kwargs.get('disp_file', None)
        self.R = kwargs.get('R', 23e-6)
        self.f_center = kwargs.get('f_center', None)
        self.f_pmp = kwargs.get('f_pmp', None)
        self.D1_manual = kwargs.get('D1_manual', None)
        self.rM_fit = kwargs.get('rM_fit', [])
        self.rM_sim = kwargs.get('rM_sim', [])
        self.debug = kwargs.get('debug', False)
        self.plot_type = kwargs.get('plot_type', 'all')

    def get_freq_modes(self,f_center=None):

        FSR0 = 1e12*23e-6/self.R
        lines = open(self.disp_file, 'r').readlines()
        self.rM = np.array([float(ll.strip().split(',')[0]) for ll in lines])  # azimuthal mode order
        self.rf = np.array([float(ll.strip().split(',')[1]) for ll in lines])  # corresponding resonance frequencies

        # find the pumped mode
        if not f_center:
            pmp_ind = np.where(np.abs(self.rf-self.f_pmp[0])<0.5*FSR0)[0]
            assert len(pmp_ind)==1, 'Pumped mode not found or multiple modes found'
            self.pmp_ind_fit = pmp_ind[0]
        else:
            pmp_ind = np.where(np.abs(self.rf-f_center)<0.5*FSR0)[0]
            assert len(pmp_ind)==1, 'Wavelength not found'
            self.pmp_ind_fit = pmp_ind[0]
        self.rM_pmp = self.rM[self.pmp_ind_fit]
        
        # plot the data
        # plt.figure()
        # plt.plot(self.rM, self.rf)
        # plt.scatter(self.rM_pmp, self.f_pmp, color='red', label='Pump Mode')
        # plt.xlabel('Azimuthal Mode Order')
        # plt.ylabel('Resonance Frequency')
        # plt.grid()
        # plt.legend()
        # plt.show()

        return self.pmp_ind_fit
    
    def get_group_index(self):
        L = 2*np.pi*self.R
        df = np.gradient(self.rf)
        self.n_eff = self.rM*const.c/(2*np.pi*self.R*self.rf)
        self.n_eff_pmp = self.n_eff[self.pmp_ind_fit]
        self.ng = const.c/(df*L)
        self.ng_pmp = self.ng[self.pmp_ind_fit]
        self.tR = L*self.ng_pmp/const.c

        # plt.figure()
        # plt.plot(self.rM, self.n_eff)
        # plt.scatter(self.rM_pmp, self.n_eff_pmp, color='red', label='Pump Mode')
        # plt.xlabel('Azimuthal Mode Order')
        # plt.ylabel('Effective Index')
        # plt.grid()
        # plt.legend()
        # plt.show()
    
    def get_dispersion(self):
        df = np.gradient(self.rf)
        d1_vg = np.gradient(self.ng)/const.c
        self.D = -(self.rf**2/const.c) * d1_vg/df

        # plt.figure()
        # plt.plot(self.rf, self.D)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel(r'Dispersion (s/$m^2$)')
        # # plt.yscale('log')
        # plt.grid()
        # plt.show()

    
    def get_integrated_dispersion(self, ind_pmp):
        # print('ind_pmp:',ind_pmp)
        dm = np.array([-2, -1, 0, 1, 2])
        drf = self.rf - self.rf[self.pmp_ind_fit]
        Dfit = np.polyfit(dm, drf[ind_pmp+dm], 2)
        self.FSR = Dfit[1]
        self.D2 = Dfit[0]*2*np.pi
        D1 = self.FSR * 2*np.pi
        mu = self.rM - self.rM[ind_pmp]
        omega = 2*np.pi*self.rf
        Dint = omega - (omega[ind_pmp] + D1 * mu)
        self.mu_0 = mu

        self.beta_2 = -self.ng_pmp/const.c*(2*Dfit[0])/Dfit[1]**2/2/np.pi
        # plot the data
        # plt.figure()
        # plt.plot(mu, Dint*1e-9/(2*np.pi))
        # plt.xlabel('Mode Order')
        # plt.ylabel('Dispersion (GHz)')
        # plt.grid()
        # plt.show()

        return mu, Dint, D1
    
    def fit_Dint_domain(self,ind0, ind_master, Dint):
        if self.rM_fit == [None, None]:
            mu_fit = [self.mu_0[0], self.mu_0[-1]]
            shift = 0
        else:
            mu_fit = self.rM_fit
            shift = ind0-ind_master
        
        mu_fit = [rm-shift for rm in mu_fit]

        M = np.arange(ind0+mu_fit[0], ind0+mu_fit[1]+1, dtype=int)

        mu_select = np.arange(mu_fit[0],mu_fit[1]+1, dtype=int)

        assert M[0]>=0 , 'Left range for mode order not correct'
        assert M[-1]<= self.rM.size, 'Right range for mode order not correct'

        M2fit = self.rM[M] - self.rM[ind0]
        Dint2fit = Dint[M]
        # print("M2 shape:",M2fit.shape, "Dint shape:",Dint2fit.shape)
        fitfun = itp.splrep(M2fit, Dint2fit)
        fit_selected = itp.splev(mu_select, fitfun)
        pmp_ind = np.argwhere(M2fit==0).flatten()[0]
        return fitfun, mu_select, fit_selected, pmp_ind


    def do_fit_sim_Dint(self,fitfun,ind0,ind_master):
        shift = ind0-ind_master
        mu2_fit = [rm-shift for rm in self.rM_sim]
        ind_sim = np.arange(mu2_fit[0], mu2_fit[1]+1, dtype=int)
        Dint_fit = itp.splev(ind_sim, fitfun)
        pmp_ind = np.argwhere(ind_sim==0).flatten()[0]
        return mu2_fit, Dint_fit, pmp_ind
    
    def getDint(self):
        self.mu_sim=[]
        self.Dint_sim=[]
        self.Dint_fit=[]
        self.pmp_ind=[]
        self.Dint = []
        self.D1 = []
        self.mu_fit=[]
        self.ind_pmp_sim=[]
        self.ind_pmp_fit=[]

        cnt_f=0
        for ff in self.f_pmp:
            if not ff == self.f_pmp[0]:
                ind_pmp = self.get_freq_modes(ff)
            else:
                ind_pmp = self.get_freq_modes()
                ind_master = ind_pmp
                self.get_group_index()
                self.get_dispersion()

            mu_, Dint_, D1_ = self.get_integrated_dispersion(ind_pmp)
            ff, mu_fit_, Dint_fit_, pmp_ind_fit_ = self.fit_Dint_domain(ind_pmp, ind_master, Dint_)
            mu_sim_, Dint_sim_, pmp_ind_sim_ = self.do_fit_sim_Dint(ff, ind_pmp, ind_master)

            # self.Dint.append(Dint)
            # self.Dint_fit.append(Dint_fit)
            # self.Dint_sim.append(Dint_sim)
            # self.D1.append(D1)
            # self.ind_pmp_sim.append(pmp_ind_sim)
            # self.ind_pmp_fit.append(pmp_ind_fit)
            # self.mu_fit.append(mu_fit)
            # self.mu_sim.append(mu_sim)

            self.Dint += [Dint_]
            self.Dint_fit += [Dint_fit_]
            self.Dint_sim += [Dint_sim_]
            self.D1 += [D1_]
            self.ind_pmp_sim += [pmp_ind_sim_]
            self.ind_pmp_fit += [pmp_ind_fit_]
            self.mu_fit += [mu_fit_]
            self.mu_sim += [mu_sim_]

            self.f_pmp[cnt_f] = self.rf[ind_pmp]
            cnt_f+=1
        
        ind0 = np.sum(self.mu_sim[0])/2
        assert ind0 == int(ind0), 'Master mode order not integer'
        ind_center = int(self.pmp_ind_fit+ind0)

        for ii in range(len(self.f_pmp)):
            self.pmp_ind += [int(-1*np.sum(self.mu_sim[ii])/2)]
        
        f_center = self.rf[ind_center]
        ind_center = self.get_freq_modes(f_center)
        mu_, Dint_, D1_ = self.get_integrated_dispersion(ind_center)
        ff_, mu_fit_, Dint_fit_, pmp_ind_fit_ = self.fit_Dint_domain(ind_center, ind_master, Dint_)
        mu_sim_, Dint_sim_, pmp_ind_sim_ = self.do_fit_sim_Dint(ff_, ind_center, ind_center-ind0)

        # self.Dint.append(Dint_)
        # self.Dint_fit.append(Dint_fit_)
        # self.Dint_sim.append(Dint_sim_)
        # self.D1.append(D1_)
        # self.f_pmp.append(f_center)
        # self.ind_pmp_sim.append(pmp_ind_sim_)
        # self.ind_pmp_fit.append(pmp_ind_fit_)
        # self.mu_fit.append(mu_fit_)
        # self.mu_sim.append(mu_sim_)

        self.f_pmp += [f_center]
        self.Dint += [Dint_]
        self.Dint_fit += [Dint_fit_]
        self.Dint_sim += [Dint_sim_]
        self.D1 += [D1_]
        self.ind_pmp_sim += [pmp_ind_sim_]
        self.ind_pmp_fit += [pmp_ind_fit_]
        self.mu_fit += [mu_fit_]
        self.mu_sim += [mu_sim_]


        mu_sim = self.mu_sim[ii]
        self.freq_fit = self.f_pmp[0]+np.arange(mu_sim[0], mu_sim[-1]+1)*self.D1[0]/(2*np.pi)

    
    # write a function to display ng, D,neff,freq freq_sim, Dint, Dint_fit, Dint_sim, FSR. print in a table with columns as parameters, description values(describing what it means) and units
    # def display_params(self):
    #     print('Results of Dispersion Analysis')
    #     print('-----------------------------------')
    #     print('Parameter\tDescription\tValue\tUnits')
    #     print('-----------------------------------')
    #     print('FSR\tFree Spectral Range\t',self.FSR,'\tHz')
    #     print('D1\tGroup Velocity Dispersion\t',self.D1[0],'\ts/m^2')
    #     print('D2\tSecond Order Dispersion\t',self.D2,'\ts/m^2')
    #     print('n_eff\tEffective Index\t',self.n_eff_pmp,'\t')
    #     print('n_g\tGroup Index\t',self.ng_pmp,'\t')
    #     print('tR\tRound Trip Time\t',self.tR,'\ts')
    #     print('-----------------------------------')
    #     print('Dispersion Analysis for Pumped Mode')
    #     print('-----------------------------------')
    #     print('Frequency\tDispersion')
    #     print('-----------------------------------')
    #     for ii in range(len(self.f_pmp)):
    #         print('%.2f THz\t%.2f GHz'%(self.f_pmp[ii]*1e-12, (self.Dint[0][self.ind_pmp_fit[ii]]-self.Dint[0])*1e-9/(2*np.pi)))
    #     print('-----------------------------------')
    #     print('Dispersion Analysis for Master Mode')
    #     print('-----------------------------------')
    #     print('Frequency\tDispersion')
    #     print('-----------------------------------')
    #     print('%.2f THz\t%.2f GHz'%(self.f_pmp[-1]*1e-12, (self.Dint[-1][self.ind_pmp_fit[-1]]-self.Dint[-1])*1e-9/(2*np.pi)))
        print('-----------------------------------')
    
    def plot_dispersion(self):
        plt.figure()

        for ii in range(len(self.f_pmp)-1):
            mu_fit = self.mu_fit[ii]
            mu_sim = self.mu_sim[ii]
            dnu_fit = np.arange(mu_fit[0], mu_fit[-1]+1)*self.D1[0]/(2*np.pi)
            dnu_sim = np.arange(mu_sim[0], mu_sim[-1]+1)*self.D1[0]/(2*np.pi)
            nu_0 = self.f_pmp[ii]
            rf = self.rf
            rf_fit = nu_0+dnu_fit
            rf_sim = nu_0+dnu_sim

            plt.plot(rf*1e-12, (self.Dint[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),'o',ms=3,label='FEM Simulation')
            plt.plot(rf_fit*1e-12, (self.Dint_fit[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),'--', ms=3,label='Fitted Dispersion')
            plt.plot(rf_sim*1e-12, (self.Dint_sim[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),label='LLE Simulation')

        # return self.fig
        plt.xlabel('Frequency (THz)', fontsize=14)
        plt.ylabel(r'$D_{int}$ (GHz)', fontsize=14)
        plt.grid()
        plt.legend()
        plt.ylim(-50,50)
        plt.title('Dispersion Analysis', fontsize=16, fontweight='bold')
        # save the figure
        # plt.savefig('dispersion.png')

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import scipy.constants as const


class Dispersion_Analysis():

    def __init__(self, **kwargs):
        self.disp_file = kwargs.get('disp_file', None)
        self.R = kwargs.get('R', 23e-6)
        self.f_center = kwargs.get('f_center', None)
        self.f_pmp = kwargs.get('f_pmp', None)
        self.D1_manual = kwargs.get('D1_manual', None)
        self.rM_fit = kwargs.get('rM_fit', [])
        self.rM_sim = kwargs.get('rM_sim', [])
        self.debug = kwargs.get('debug', False)
        self.plot_type = kwargs.get('plot_type', 'all')

    def get_freq_modes(self,f_center=None):

        FSR0 = 1e12*23e-6/self.R
        lines = open(self.disp_file, 'r').readlines()
        self.rM = np.array([float(ll.strip().split(',')[0]) for ll in lines])  # azimuthal mode order
        self.rf = np.array([float(ll.strip().split(',')[1]) for ll in lines])  # corresponding resonance frequencies

        # find the pumped mode
        if not f_center:
            pmp_ind = np.where(np.abs(self.rf-self.f_pmp[0])<0.5*FSR0)[0]
            assert len(pmp_ind)==1, 'Pumped mode not found or multiple modes found'
            self.pmp_ind_fit = pmp_ind[0]
        else:
            pmp_ind = np.where(np.abs(self.rf-f_center)<0.5*FSR0)[0]
            assert len(pmp_ind)==1, 'Wavelength not found'
            self.pmp_ind_fit = pmp_ind[0]
        self.rM_pmp = self.rM[self.pmp_ind_fit]
        
        # plot the data
        # plt.figure()
        # plt.plot(self.rM, self.rf)
        # plt.scatter(self.rM_pmp, self.f_pmp, color='red', label='Pump Mode')
        # plt.xlabel('Azimuthal Mode Order')
        # plt.ylabel('Resonance Frequency')
        # plt.grid()
        # plt.legend()
        # plt.show()

        return self.pmp_ind_fit
    
    def get_group_index(self):
        L = 2*np.pi*self.R
        df = np.gradient(self.rf)
        self.n_eff = self.rM*const.c/(2*np.pi*self.R*self.rf)
        self.n_eff_pmp = self.n_eff[self.pmp_ind_fit]
        self.ng = const.c/(df*L)
        self.ng_pmp = self.ng[self.pmp_ind_fit]
        self.tR = L*self.ng_pmp/const.c

        # plt.figure()
        # plt.plot(self.rM, self.n_eff)
        # plt.scatter(self.rM_pmp, self.n_eff_pmp, color='red', label='Pump Mode')
        # plt.xlabel('Azimuthal Mode Order')
        # plt.ylabel('Effective Index')
        # plt.grid()
        # plt.legend()
        # plt.show()
    
    def get_dispersion(self):
        df = np.gradient(self.rf)
        d1_vg = np.gradient(self.ng)/const.c
        self.D = -(self.rf**2/const.c) * d1_vg/df

        # plt.figure()
        # plt.plot(self.rf, self.D)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel(r'Dispersion (s/$m^2$)')
        # # plt.yscale('log')
        # plt.grid()
        # plt.show()

    
    def get_integrated_dispersion(self, ind_pmp):
        # print('ind_pmp:',ind_pmp)
        dm = np.array([-2, -1, 0, 1, 2])
        drf = self.rf - self.rf[self.pmp_ind_fit]
        Dfit = np.polyfit(dm, drf[ind_pmp+dm], 2)
        self.FSR = Dfit[1]
        self.D2 = Dfit[0]*2*np.pi
        D1 = self.FSR * 2*np.pi
        mu = self.rM - self.rM[ind_pmp]
        omega = 2*np.pi*self.rf
        Dint = omega - (omega[ind_pmp] + D1 * mu)
        self.mu_0 = mu

        self.beta_2 = -self.ng_pmp/const.c*(2*Dfit[0])/Dfit[1]**2/2/np.pi
        # plot the data
        # plt.figure()
        # plt.plot(mu, Dint*1e-9/(2*np.pi))
        # plt.xlabel('Mode Order')
        # plt.ylabel('Dispersion (GHz)')
        # plt.grid()
        # plt.show()

        return mu, Dint, D1
    
    def fit_Dint_domain(self,ind0, ind_master, Dint):
        if self.rM_fit == [None, None]:
            mu_fit = [self.mu_0[0], self.mu_0[-1]]
            shift = 0
        else:
            mu_fit = self.rM_fit
            shift = ind0-ind_master
        
        mu_fit = [rm-shift for rm in mu_fit]

        M = np.arange(ind0+mu_fit[0], ind0+mu_fit[1]+1, dtype=int)

        mu_select = np.arange(mu_fit[0],mu_fit[1]+1, dtype=int)

        assert M[0]>=0 , 'Left range for mode order not correct'
        assert M[-1]<= self.rM.size, 'Right range for mode order not correct'

        M2fit = self.rM[M] - self.rM[ind0]
        Dint2fit = Dint[M]
        # print("M2 shape:",M2fit.shape, "Dint shape:",Dint2fit.shape)
        fitfun = itp.splrep(M2fit, Dint2fit)
        fit_selected = itp.splev(mu_select, fitfun)
        pmp_ind = np.argwhere(M2fit==0).flatten()[0]
        return fitfun, mu_select, fit_selected, pmp_ind


    def do_fit_sim_Dint(self,fitfun,ind0,ind_master):
        shift = ind0-ind_master
        mu2_fit = [rm-shift for rm in self.rM_sim]
        ind_sim = np.arange(mu2_fit[0], mu2_fit[1]+1, dtype=int)
        Dint_fit = itp.splev(ind_sim, fitfun)
        pmp_ind = np.argwhere(ind_sim==0).flatten()[0]
        return mu2_fit, Dint_fit, pmp_ind
    
    def getDint(self):
        self.mu_sim=[]
        self.Dint_sim=[]
        self.Dint_fit=[]
        self.pmp_ind=[]
        self.Dint = []
        self.D1 = []
        self.mu_fit=[]
        self.ind_pmp_sim=[]
        self.ind_pmp_fit=[]

        cnt_f=0
        for ff in self.f_pmp:
            if not ff == self.f_pmp[0]:
                ind_pmp = self.get_freq_modes(ff)
            else:
                ind_pmp = self.get_freq_modes()
                ind_master = ind_pmp
                self.get_group_index()
                self.get_dispersion()

            mu_, Dint_, D1_ = self.get_integrated_dispersion(ind_pmp)
            ff, mu_fit_, Dint_fit_, pmp_ind_fit_ = self.fit_Dint_domain(ind_pmp, ind_master, Dint_)
            mu_sim_, Dint_sim_, pmp_ind_sim_ = self.do_fit_sim_Dint(ff, ind_pmp, ind_master)

            # self.Dint.append(Dint)
            # self.Dint_fit.append(Dint_fit)
            # self.Dint_sim.append(Dint_sim)
            # self.D1.append(D1)
            # self.ind_pmp_sim.append(pmp_ind_sim)
            # self.ind_pmp_fit.append(pmp_ind_fit)
            # self.mu_fit.append(mu_fit)
            # self.mu_sim.append(mu_sim)

            self.Dint += [Dint_]
            self.Dint_fit += [Dint_fit_]
            self.Dint_sim += [Dint_sim_]
            self.D1 += [D1_]
            self.ind_pmp_sim += [pmp_ind_sim_]
            self.ind_pmp_fit += [pmp_ind_fit_]
            self.mu_fit += [mu_fit_]
            self.mu_sim += [mu_sim_]

            self.f_pmp[cnt_f] = self.rf[ind_pmp]
            cnt_f+=1
        
        ind0 = np.sum(self.mu_sim[0])/2
        assert ind0 == int(ind0), 'Master mode order not integer'
        ind_center = int(self.pmp_ind_fit+ind0)

        for ii in range(len(self.f_pmp)):
            self.pmp_ind += [int(-1*np.sum(self.mu_sim[ii])/2)]
        
        f_center = self.rf[ind_center]
        ind_center = self.get_freq_modes(f_center)
        mu_, Dint_, D1_ = self.get_integrated_dispersion(ind_center)
        ff_, mu_fit_, Dint_fit_, pmp_ind_fit_ = self.fit_Dint_domain(ind_center, ind_master, Dint_)
        mu_sim_, Dint_sim_, pmp_ind_sim_ = self.do_fit_sim_Dint(ff_, ind_center, ind_center-ind0)

        # self.Dint.append(Dint_)
        # self.Dint_fit.append(Dint_fit_)
        # self.Dint_sim.append(Dint_sim_)
        # self.D1.append(D1_)
        # self.f_pmp.append(f_center)
        # self.ind_pmp_sim.append(pmp_ind_sim_)
        # self.ind_pmp_fit.append(pmp_ind_fit_)
        # self.mu_fit.append(mu_fit_)
        # self.mu_sim.append(mu_sim_)

        self.f_pmp += [f_center]
        self.Dint += [Dint_]
        self.Dint_fit += [Dint_fit_]
        self.Dint_sim += [Dint_sim_]
        self.D1 += [D1_]
        self.ind_pmp_sim += [pmp_ind_sim_]
        self.ind_pmp_fit += [pmp_ind_fit_]
        self.mu_fit += [mu_fit_]
        self.mu_sim += [mu_sim_]


        mu_sim = self.mu_sim[ii]
        self.freq_fit = self.f_pmp[0]+np.arange(mu_sim[0], mu_sim[-1]+1)*self.D1[0]/(2*np.pi)

    
    # write a function to display ng, D,neff,freq freq_sim, Dint, Dint_fit, Dint_sim, FSR. print in a table with columns as parameters, description values(describing what it means) and units
    # def display_params(self):
    #     print('Results of Dispersion Analysis')
    #     print('-----------------------------------')
    #     print('Parameter\tDescription\tValue\tUnits')
    #     print('-----------------------------------')
    #     print('FSR\tFree Spectral Range\t',self.FSR,'\tHz')
    #     print('D1\tGroup Velocity Dispersion\t',self.D1[0],'\ts/m^2')
    #     print('D2\tSecond Order Dispersion\t',self.D2,'\ts/m^2')
    #     print('n_eff\tEffective Index\t',self.n_eff_pmp,'\t')
    #     print('n_g\tGroup Index\t',self.ng_pmp,'\t')
    #     print('tR\tRound Trip Time\t',self.tR,'\ts')
    #     print('-----------------------------------')
    #     print('Dispersion Analysis for Pumped Mode')
    #     print('-----------------------------------')
    #     print('Frequency\tDispersion')
    #     print('-----------------------------------')
    #     for ii in range(len(self.f_pmp)):
    #         print('%.2f THz\t%.2f GHz'%(self.f_pmp[ii]*1e-12, (self.Dint[0][self.ind_pmp_fit[ii]]-self.Dint[0])*1e-9/(2*np.pi)))
    #     print('-----------------------------------')
    #     print('Dispersion Analysis for Master Mode')
    #     print('-----------------------------------')
    #     print('Frequency\tDispersion')
    #     print('-----------------------------------')
    #     print('%.2f THz\t%.2f GHz'%(self.f_pmp[-1]*1e-12, (self.Dint[-1][self.ind_pmp_fit[-1]]-self.Dint[-1])*1e-9/(2*np.pi)))
        print('-----------------------------------')
    
    def plot_dispersion(self):
        plt.figure()

        for ii in range(len(self.f_pmp)-1):
            mu_fit = self.mu_fit[ii]
            mu_sim = self.mu_sim[ii]
            dnu_fit = np.arange(mu_fit[0], mu_fit[-1]+1)*self.D1[0]/(2*np.pi)
            dnu_sim = np.arange(mu_sim[0], mu_sim[-1]+1)*self.D1[0]/(2*np.pi)
            nu_0 = self.f_pmp[ii]
            rf = self.rf
            rf_fit = nu_0+dnu_fit
            rf_sim = nu_0+dnu_sim

            plt.plot(rf*1e-12, (self.Dint[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),'o',ms=3,label='FEM Simulation')
            plt.plot(rf_fit*1e-12, (self.Dint_fit[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),'--', ms=3,label='Fitted Dispersion')
            plt.plot(rf_sim*1e-12, (self.Dint_sim[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),label='LLE Simulation')

        # return self.fig
        plt.xlabel('Frequency (THz)', fontsize=14)
        plt.ylabel(r'$D_{int}$ (GHz)', fontsize=14)
        plt.grid()
        plt.legend()
        plt.ylim(-50,50)
        plt.title('Dispersion Analysis', fontsize=16, fontweight='bold')
        # save the figure
        # plt.savefig('dispersion.png')
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import scipy.constants as const


class Dispersion_Analysis():

    def __init__(self, **kwargs):
        self.disp_file = kwargs.get('disp_file', None)
        self.R = kwargs.get('R', 23e-6)
        self.f_center = kwargs.get('f_center', None)
        self.f_pmp = kwargs.get('f_pmp', None)
        self.D1_manual = kwargs.get('D1_manual', None)
        self.rM_fit = kwargs.get('rM_fit', [])
        self.rM_sim = kwargs.get('rM_sim', [])
        self.debug = kwargs.get('debug', False)
        self.plot_type = kwargs.get('plot_type', 'all')

    def get_freq_modes(self,f_center=None):

        FSR0 = 1e12*23e-6/self.R
        lines = open(self.disp_file, 'r').readlines()
        self.rM = np.array([float(ll.strip().split(',')[0]) for ll in lines])  # azimuthal mode order
        self.rf = np.array([float(ll.strip().split(',')[1]) for ll in lines])  # corresponding resonance frequencies

        # find the pumped mode
        if not f_center:
            pmp_ind = np.where(np.abs(self.rf-self.f_pmp[0])<0.5*FSR0)[0]
            assert len(pmp_ind)==1, 'Pumped mode not found or multiple modes found'
            self.pmp_ind_fit = pmp_ind[0]
        else:
            pmp_ind = np.where(np.abs(self.rf-f_center)<0.5*FSR0)[0]
            assert len(pmp_ind)==1, 'Wavelength not found'
            self.pmp_ind_fit = pmp_ind[0]
        self.rM_pmp = self.rM[self.pmp_ind_fit]
        
        # plot the data
        # plt.figure()
        # plt.plot(self.rM, self.rf)
        # plt.scatter(self.rM_pmp, self.f_pmp, color='red', label='Pump Mode')
        # plt.xlabel('Azimuthal Mode Order')
        # plt.ylabel('Resonance Frequency')
        # plt.grid()
        # plt.legend()
        # plt.show()

        return self.pmp_ind_fit
    
    def get_group_index(self):
        L = 2*np.pi*self.R
        df = np.gradient(self.rf)
        self.n_eff = self.rM*const.c/(2*np.pi*self.R*self.rf)
        self.n_eff_pmp = self.n_eff[self.pmp_ind_fit]
        self.ng = const.c/(df*L)
        self.ng_pmp = self.ng[self.pmp_ind_fit]
        self.tR = L*self.ng_pmp/const.c

        # plt.figure()
        # plt.plot(self.rM, self.n_eff)
        # plt.scatter(self.rM_pmp, self.n_eff_pmp, color='red', label='Pump Mode')
        # plt.xlabel('Azimuthal Mode Order')
        # plt.ylabel('Effective Index')
        # plt.grid()
        # plt.legend()
        # plt.show()
    
    def get_dispersion(self):
        df = np.gradient(self.rf)
        d1_vg = np.gradient(self.ng)/const.c
        self.D = -(self.rf**2/const.c) * d1_vg/df

        # plt.figure()
        # plt.plot(self.rf, self.D)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel(r'Dispersion (s/$m^2$)')
        # # plt.yscale('log')
        # plt.grid()
        # plt.show()

    
    def get_integrated_dispersion(self, ind_pmp):
        # print('ind_pmp:',ind_pmp)
        dm = np.array([-2, -1, 0, 1, 2])
        drf = self.rf - self.rf[self.pmp_ind_fit]
        Dfit = np.polyfit(dm, drf[ind_pmp+dm], 2)
        self.FSR = Dfit[1]
        self.D2 = Dfit[0]*2*np.pi
        D1 = self.FSR * 2*np.pi
        mu = self.rM - self.rM[ind_pmp]
        omega = 2*np.pi*self.rf
        Dint = omega - (omega[ind_pmp] + D1 * mu)
        self.mu_0 = mu

        self.beta_2 = -self.ng_pmp/const.c*(2*Dfit[0])/Dfit[1]**2/2/np.pi
        # plot the data
        # plt.figure()
        # plt.plot(mu, Dint*1e-9/(2*np.pi))
        # plt.xlabel('Mode Order')
        # plt.ylabel('Dispersion (GHz)')
        # plt.grid()
        # plt.show()

        return mu, Dint, D1
    
    def fit_Dint_domain(self,ind0, ind_master, Dint):
        if self.rM_fit == [None, None]:
            mu_fit = [self.mu_0[0], self.mu_0[-1]]
            shift = 0
        else:
            mu_fit = self.rM_fit
            shift = ind0-ind_master
        
        mu_fit = [rm-shift for rm in mu_fit]

        M = np.arange(ind0+mu_fit[0], ind0+mu_fit[1]+1, dtype=int)

        mu_select = np.arange(mu_fit[0],mu_fit[1]+1, dtype=int)

        assert M[0]>=0 , 'Left range for mode order not correct'
        assert M[-1]<= self.rM.size, 'Right range for mode order not correct'

        M2fit = self.rM[M] - self.rM[ind0]
        Dint2fit = Dint[M]
        # print("M2 shape:",M2fit.shape, "Dint shape:",Dint2fit.shape)
        fitfun = itp.splrep(M2fit, Dint2fit)
        fit_selected = itp.splev(mu_select, fitfun)
        pmp_ind = np.argwhere(M2fit==0).flatten()[0]
        return fitfun, mu_select, fit_selected, pmp_ind


    def do_fit_sim_Dint(self,fitfun,ind0,ind_master):
        shift = ind0-ind_master
        mu2_fit = [rm-shift for rm in self.rM_sim]
        ind_sim = np.arange(mu2_fit[0], mu2_fit[1]+1, dtype=int)
        Dint_fit = itp.splev(ind_sim, fitfun)
        pmp_ind = np.argwhere(ind_sim==0).flatten()[0]
        return mu2_fit, Dint_fit, pmp_ind
    
    def getDint(self):
        self.mu_sim=[]
        self.Dint_sim=[]
        self.Dint_fit=[]
        self.pmp_ind=[]
        self.Dint = []
        self.D1 = []
        self.mu_fit=[]
        self.ind_pmp_sim=[]
        self.ind_pmp_fit=[]

        cnt_f=0
        for ff in self.f_pmp:
            if not ff == self.f_pmp[0]:
                ind_pmp = self.get_freq_modes(ff)
            else:
                ind_pmp = self.get_freq_modes()
                ind_master = ind_pmp
                self.get_group_index()
                self.get_dispersion()

            mu_, Dint_, D1_ = self.get_integrated_dispersion(ind_pmp)
            ff, mu_fit_, Dint_fit_, pmp_ind_fit_ = self.fit_Dint_domain(ind_pmp, ind_master, Dint_)
            mu_sim_, Dint_sim_, pmp_ind_sim_ = self.do_fit_sim_Dint(ff, ind_pmp, ind_master)

            # self.Dint.append(Dint)
            # self.Dint_fit.append(Dint_fit)
            # self.Dint_sim.append(Dint_sim)
            # self.D1.append(D1)
            # self.ind_pmp_sim.append(pmp_ind_sim)
            # self.ind_pmp_fit.append(pmp_ind_fit)
            # self.mu_fit.append(mu_fit)
            # self.mu_sim.append(mu_sim)

            self.Dint += [Dint_]
            self.Dint_fit += [Dint_fit_]
            self.Dint_sim += [Dint_sim_]
            self.D1 += [D1_]
            self.ind_pmp_sim += [pmp_ind_sim_]
            self.ind_pmp_fit += [pmp_ind_fit_]
            self.mu_fit += [mu_fit_]
            self.mu_sim += [mu_sim_]

            self.f_pmp[cnt_f] = self.rf[ind_pmp]
            cnt_f+=1
        
        ind0 = np.sum(self.mu_sim[0])/2
        assert ind0 == int(ind0), 'Master mode order not integer'
        ind_center = int(self.pmp_ind_fit+ind0)

        for ii in range(len(self.f_pmp)):
            self.pmp_ind += [int(-1*np.sum(self.mu_sim[ii])/2)]
        
        f_center = self.rf[ind_center]
        ind_center = self.get_freq_modes(f_center)
        mu_, Dint_, D1_ = self.get_integrated_dispersion(ind_center)
        ff_, mu_fit_, Dint_fit_, pmp_ind_fit_ = self.fit_Dint_domain(ind_center, ind_master, Dint_)
        mu_sim_, Dint_sim_, pmp_ind_sim_ = self.do_fit_sim_Dint(ff_, ind_center, ind_center-ind0)

        # self.Dint.append(Dint_)
        # self.Dint_fit.append(Dint_fit_)
        # self.Dint_sim.append(Dint_sim_)
        # self.D1.append(D1_)
        # self.f_pmp.append(f_center)
        # self.ind_pmp_sim.append(pmp_ind_sim_)
        # self.ind_pmp_fit.append(pmp_ind_fit_)
        # self.mu_fit.append(mu_fit_)
        # self.mu_sim.append(mu_sim_)

        self.f_pmp += [f_center]
        self.Dint += [Dint_]
        self.Dint_fit += [Dint_fit_]
        self.Dint_sim += [Dint_sim_]
        self.D1 += [D1_]
        self.ind_pmp_sim += [pmp_ind_sim_]
        self.ind_pmp_fit += [pmp_ind_fit_]
        self.mu_fit += [mu_fit_]
        self.mu_sim += [mu_sim_]


        mu_sim = self.mu_sim[ii]
        self.freq_fit = self.f_pmp[0]+np.arange(mu_sim[0], mu_sim[-1]+1)*self.D1[0]/(2*np.pi)

    
    # write a function to display ng, D,neff,freq freq_sim, Dint, Dint_fit, Dint_sim, FSR. print in a table with columns as parameters, description values(describing what it means) and units
    # def display_params(self):
    #     print('Results of Dispersion Analysis')
    #     print('-----------------------------------')
    #     print('Parameter\tDescription\tValue\tUnits')
    #     print('-----------------------------------')
    #     print('FSR\tFree Spectral Range\t',self.FSR,'\tHz')
    #     print('D1\tGroup Velocity Dispersion\t',self.D1[0],'\ts/m^2')
    #     print('D2\tSecond Order Dispersion\t',self.D2,'\ts/m^2')
    #     print('n_eff\tEffective Index\t',self.n_eff_pmp,'\t')
    #     print('n_g\tGroup Index\t',self.ng_pmp,'\t')
    #     print('tR\tRound Trip Time\t',self.tR,'\ts')
    #     print('-----------------------------------')
    #     print('Dispersion Analysis for Pumped Mode')
    #     print('-----------------------------------')
    #     print('Frequency\tDispersion')
    #     print('-----------------------------------')
    #     for ii in range(len(self.f_pmp)):
    #         print('%.2f THz\t%.2f GHz'%(self.f_pmp[ii]*1e-12, (self.Dint[0][self.ind_pmp_fit[ii]]-self.Dint[0])*1e-9/(2*np.pi)))
    #     print('-----------------------------------')
    #     print('Dispersion Analysis for Master Mode')
    #     print('-----------------------------------')
    #     print('Frequency\tDispersion')
    #     print('-----------------------------------')
    #     print('%.2f THz\t%.2f GHz'%(self.f_pmp[-1]*1e-12, (self.Dint[-1][self.ind_pmp_fit[-1]]-self.Dint[-1])*1e-9/(2*np.pi)))
        print('-----------------------------------')
    
    def plot_dispersion(self):
        plt.figure()

        for ii in range(len(self.f_pmp)-1):
            mu_fit = self.mu_fit[ii]
            mu_sim = self.mu_sim[ii]
            dnu_fit = np.arange(mu_fit[0], mu_fit[-1]+1)*self.D1[0]/(2*np.pi)
            dnu_sim = np.arange(mu_sim[0], mu_sim[-1]+1)*self.D1[0]/(2*np.pi)
            nu_0 = self.f_pmp[ii]
            rf = self.rf
            rf_fit = nu_0+dnu_fit
            rf_sim = nu_0+dnu_sim

            plt.plot(rf*1e-12, (self.Dint[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),'o',ms=3,label='FEM Simulation')
            plt.plot(rf_fit*1e-12, (self.Dint_fit[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),'--', ms=3,label='Fitted Dispersion')
            plt.plot(rf_sim*1e-12, (self.Dint_sim[0]-self.Dint[0][self.ind_pmp_fit[ii]])*1e-9/(2*np.pi),label='LLE Simulation')

        # return self.fig
        plt.xlabel('Frequency (THz)', fontsize=14)
        plt.ylabel(r'$D_{int}$ (GHz)', fontsize=14)
        plt.grid()
        plt.legend()
        plt.ylim(-50,50)
        plt.title('Dispersion Analysis', fontsize=16, fontweight='bold')
        # save the figure
        # plt.savefig('dispersion.png')


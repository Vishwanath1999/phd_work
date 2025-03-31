<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 68f4d66462f7ec52f9d569e23942b3c376910abe
# %%
import numpy as np
from my_disp import Dispersion_Analysis as Disp
import scipy.constants as const
class LLETest:
    def __init__(self, **kwargs):
        self.res = kwargs.get('res', {})
        self.sim = kwargs.get('sim', {})

        print(self.res)
        print(self.sim)

        self.debug = kwargs.get('debug', False)
        if not 'D1_manual' in self.sim.keys():
            self.res['D1_manual'] = None
        if not 'f_pmp' in self.sim.keys():
            self.res['f_pmp'] = const.c/self.res['lambda_pmp']
        if not 'mu_fit' in self.sim.keys():
            self.res['mu_fit'] = [None, None]
        if not 'num_probe' in self.sim.keys():
            self.res['num_probe'] = 1000
        
    def analyze(self):

        self._analyze = Disp(disp_file=res['disp_file'], R=self.res['R'],\
                              f_pmp=self.sim['f_pmp'], D1_manual=self.sim['D1_manual'],\
                                  rM_fit=self.sim['mu_fit'], rM_sim=self.sim['mu_sim'],\
                                      debug=self.debug)
        self._analyze.getDint()
        # self._analyze.display_params()
        # fig = self._analyze.plot_dispersion()

        self.
        

# %%
res = dict(
        R=23e-6, 
        Qi=1e6, 
        Qc=1e6, 
        gamma=3.2, 
        disp_file='RW1000_H430.csv',
)

sim = dict(
    Pin=[240e-3], 
    f_pmp=[283e12],
    phi_pmp=[0], 
    del_omega=[None], 
    Tscan=0.7e6,
    mu_sim=[-220, 220],
    mu_fit=[None, None],
    del_omega_init= 1e9 * 2 * np.pi,
    del_omega_end= -6.5e9 * 2 * np.pi,
    num_probe = 5000, 
    D1_manual = None
)

lle = LLETest(res=res, sim=sim)

lle.analyze()

# %%
=======
<<<<<<< HEAD
# %%
import numpy as np
from my_disp import Dispersion_Analysis as Disp
import scipy.constants as const
class LLETest:
    def __init__(self, **kwargs):
        self.res = kwargs.get('res', {})
        self.sim = kwargs.get('sim', {})

        print(self.res)
        print(self.sim)

        self.debug = kwargs.get('debug', False)
        if not 'D1_manual' in self.sim.keys():
            self.res['D1_manual'] = None
        if not 'f_pmp' in self.sim.keys():
            self.res['f_pmp'] = const.c/self.res['lambda_pmp']
        if not 'mu_fit' in self.sim.keys():
            self.res['mu_fit'] = [None, None]
        if not 'num_probe' in self.sim.keys():
            self.res['num_probe'] = 1000
        
    def analyze(self):

        self._analyze = Disp(disp_file=res['disp_file'], R=self.res['R'],\
                              f_pmp=self.sim['f_pmp'], D1_manual=self.sim['D1_manual'],\
                                  rM_fit=self.sim['mu_fit'], rM_sim=self.sim['mu_sim'],\
                                      debug=self.debug)
        self._analyze.getDint()
        self._analyze.display_params()
        fig = self._analyze.plot_dispersion()

# %%
res = dict(
        R=23e-6, 
        Qi=1e6, 
        Qc=1e6, 
        gamma=3.2, 
        disp_file='RW1000_H430.csv',
)

sim = dict(
    Pin=[240e-3], 
    f_pmp=[283e12],
    phi_pmp=[0], 
    del_omega=[None], 
    Tscan=0.7e6,
    mu_sim=[-220, 220],
    mu_fit=[None, None],
    del_omega_init= 1e9 * 2 * np.pi,
    del_omega_end= -6.5e9 * 2 * np.pi,
    num_probe = 5000, 
    D1_manual = None
)

lle = LLETest(res=res, sim=sim)

lle.analyze()
print(lle._analyze.pmp_ind_fit)
=======
# %%
import numpy as np
from my_disp import Dispersion_Analysis as Disp
import scipy.constants as const
class LLETest:
    def __init__(self, **kwargs):
        self.res = kwargs.get('res', {})
        self.sim = kwargs.get('sim', {})

        print(self.res)
        print(self.sim)

        self.debug = kwargs.get('debug', False)
        if not 'D1_manual' in self.sim.keys():
            self.res['D1_manual'] = None
        if not 'f_pmp' in self.sim.keys():
            self.res['f_pmp'] = const.c/self.res['lambda_pmp']
        if not 'mu_fit' in self.sim.keys():
            self.res['mu_fit'] = [None, None]
        if not 'num_probe' in self.sim.keys():
            self.res['num_probe'] = 1000
        
    def analyze(self):

        self._analyze = Disp(disp_file=res['disp_file'], R=self.res['R'],\
                              f_pmp=self.sim['f_pmp'], D1_manual=self.sim['D1_manual'],\
                                  rM_fit=self.sim['mu_fit'], rM_sim=self.sim['mu_sim'],\
                                      debug=self.debug)
        self._analyze.getDint()
        self._analyze.display_params()
        fig = self._analyze.plot_dispersion()

# %%
res = dict(
        R=23e-6, 
        Qi=1e6, 
        Qc=1e6, 
        gamma=3.2, 
        disp_file='RW1000_H430.csv',
)

sim = dict(
    Pin=[240e-3], 
    f_pmp=[283e12],
    phi_pmp=[0], 
    del_omega=[None], 
    Tscan=0.7e6,
    mu_sim=[-220, 220],
    mu_fit=[None, None],
    del_omega_init= 1e9 * 2 * np.pi,
    del_omega_end= -6.5e9 * 2 * np.pi,
    num_probe = 5000, 
    D1_manual = None
)

lle = LLETest(res=res, sim=sim)

lle.analyze()
print(lle._analyze.pmp_ind_fit)
>>>>>>> 68f4d66462f7ec52f9d569e23942b3c376910abe
print(lle._analyze.FSR)
>>>>>>> 6faa7c03e886ec4cecd74317c11f630c5ff56af4

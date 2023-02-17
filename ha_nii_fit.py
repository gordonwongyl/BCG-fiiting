from fit_routine import WLAX, Lines, lines, hamod1, hamod2, chisq, c, z, niil_wratio, niir_wratio
from typing import Callable, Iterable, List, Tuple, Union, Optional
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy

class Ha_NII(Lines):
    def __init__(self, name: str, rest, crange, vrange, cube: NDArray, varc: NDArray, contspec: NDArray):
        super().__init__(name, rest, crange, vrange, cube, varc, contspec)
        self.CUBE = cube
        self.fitcube = np.zeros((14, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((8 ,self.cube_x, self.cube_y))
        self.mod1_fitcube = np.zeros((14, self.cube_x, self.cube_y))
        self.mod1_fiterrcube = np.zeros((8 ,self.cube_x, self.cube_y))
        self.mod2_fitcube = np.zeros((14, self.cube_x, self.cube_y))
        self.mod2_fiterrcube = np.zeros((8 ,self.cube_x, self.cube_y))

    def plot_spe(self, ax_in: plt.Axes, i: int, j: int) -> None:

        if self.mod1_fitcube is None or self.mod2_fitcube is None:
            return print("No data")
 
        lranges = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])
        fit_spec, errspec = self.get_fit_spaxel(i, j)
        popt1 = self.mod1_fitcube[4:8, i, j]
        snr_nii = self.mod1_fitcube[8, i, j]
        snr_ha = self.mod1_fitcube[9, i, j]
        chi1 = self.mod1_fitcube[10, i, j]

        popt2 = self.mod2_fitcube[0:8, i, j]
        snr_nii_1 = self.mod2_fitcube[8, i, j]
        snr_ha_1 = self.mod2_fitcube[9, i, j]
        snr_nii_2 = self.mod2_fitcube[10, i, j]
        snr_ha_2 = self.mod2_fitcube[11, i, j]
        vel_snr = self.mod2_fitcube[12, i, j]
        chi2 = self.mod2_fitcube[13, i, j]
        nii_ratio = popt2[0]/popt2[4] if popt2[0] < popt2[4] else popt2[4]/popt2[0]
        ha_ratio = popt2[1]/popt2[5] if popt2[1] < popt2[5] else popt2[5]/popt2[1]
         
        #plot
        #ax_in[0].step(self.wlax, fit_spec, where='mid', color='black')
        ax_in[0].step(self.wlax[self.mask], fit_spec[self.mask], where='mid', color='#1f77b4')
        ax_in[0].plot(self.wlax[self.mask], hamod1(self.wlax[self.mask], *popt1), color='Orange')
        ax_in[0].axvline(popt1[2], color='orange', label='central wavelength')
        ax_in[0].axvline(popt1[2]*niir_wratio, color='orange')
        ax_in[0].axvline(popt1[2]*niil_wratio, color='orange')
        ax_in[0].set_title(f"Sample fit with one velocity component")
        ax_in[0].set(xlabel='Wavelength(Angstrom)', ylabel='F (10**-20 Angstrom-1 cm-2 ergs-1)', xlim=(6600,6680))
        ax_in[0].legend()

        
        
        #compared to hamod2
        #ax_in[1].step(self.wlax, fit_spec, where='mid', color='black')
        ax_in[1].step(self.wlax[self.mask], fit_spec[self.mask], where='mid', color='#1f77b4')
        ax_in[1].plot(self.wlax[self.mask], hamod2(self.wlax[self.mask], *popt2), color='Orange')
        ax_in[1].plot(self.wlax[self.mask], hamod1(self.wlax[self.mask], *popt2[0:4]), color='red')
        ax_in[1].plot(self.wlax[self.mask], hamod1(self.wlax[self.mask], *popt2[4:8]), color='green')
        ax_in[1].axvline(popt2[2], color='red', label='lower velocity component')
        ax_in[1].axvline(popt2[2]*niir_wratio, color='red')
        ax_in[1].axvline(popt2[2]*niil_wratio, color='red')
        ax_in[1].axvline(popt2[6], color='green', label='higher velocity component')
        ax_in[1].axvline(popt2[6]*niir_wratio, color='green')
        ax_in[1].axvline(popt2[6]*niil_wratio, color='green')
        ax_in[1].set_title(f"Sample fit with two velocity components")
        ax_in[1].set(xlabel='Wavelength(Angstrom)', ylabel='F (10**-20 Angstrom-1 cm-2 ergs-1)', xlim=(6600,6680))
        ax_in[1].legend()

        # base_cont_lvl = self.get_cont_lvl(self.basespec)
        # line_cont_lvl = self.get_cont_lvl(self.spec)
        # aC = (line_cont_lvl/base_cont_lvl) * self.basespec

        #plot
        # ax_in[1].step(self.wlax, fit_spec, where='mid', color='black', label='Continuum range (Smask)')
        # ax_in[1].step(self.wlax[lranges], fit_spec[lranges], where='mid', label = 'Line range')
        # ax_in[1].plot(self.wlax[lranges], hamod1(self.wlax[lranges], *popt), label='Line fitting')
        # #ax_in[0].axhline(y=np.median(fit_spec[lranges]), color='r', label='Median around line ranges')
        # ax_in[1].set_title(f"(i,j): ({j+1}, {i+1}), Residual Spectrum (R)")
        # ax_in[1].legend(loc='upper left')
        # ax_in[1].set(xlabel='Wavelength(Angstrom)', ylabel='F (10**-20 Angstrom-1 cm-2 ergs-1)')
        

        # ax_in[0].step(self.wlax, self.spec, where='mid', color='black', label='Continuum range (Smask)')
        # ax_in[0].step(self.wlax[lranges], self.spec[lranges], where='mid', label='Line range')
        # ax_in[0].step(self.wlax, aC, where='mid', label='scaled template baseline (aC)')
        # ax_in[0].set(xlabel='Wavelength(Angstrom)', ylabel='F (10**-20 Angstrom-1 cm-2 ergs-1)', ylim=(1300,2000)) 
        # ax_in[0].legend(loc='upper left')
        # ax_in[0].set_title(f"Original Spectrum (S) at ({j+1},{i+1}) scale factor (alpha) = {(line_cont_lvl/base_cont_lvl):.2f}")
        # ...

    def load_2fitcubes(self, MOD1_FITCUBE_DIR: str, MOD2_FITCUBE_DIR: str) -> None:
        mod1fithdul = fits.open(MOD1_FITCUBE_DIR)
        mod2fithdul = fits.open(MOD2_FITCUBE_DIR) 
        

        for k in range(self.fitcube.shape[0]):
            for j in range(self.cube_x):
                for i in range(self.cube_y):
                    self.mod1_fitcube[k,j,i] = mod1fithdul[k].data[j,i]
                    self.mod2_fitcube[k,j,i] = mod2fithdul[k].data[j,i]

        for k in range(self.fitcube.shape[0], len(mod1fithdul)):
            for j in range(self.cube_x):
                for i in range(self.cube_y):
                    self.mod1_fiterrcube[k-self.fitcube.shape[0],j,i] = mod1fithdul[k].data[j,i]
                    self.mod2_fiterrcube[k-self.fitcube.shape[0],j,i] = mod2fithdul[k].data[j,i]

    def select(self, i, j):
        self.fitcube = np.zeros((14, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((8 ,self.cube_x, self.cube_y))
        
        popt1 = self.mod1_fitcube[4:8, i, j]
        snr_nii = self.mod1_fitcube[8, i, j]
        snr_ha = self.mod1_fitcube[9, i, j]
        chi1 = self.mod1_fitcube[10, i, j]

        popt2 = self.mod2_fitcube[0:8, i, j]
        snr_nii_1 = self.mod2_fitcube[8, i, j].copy()
        snr_ha_1 = self.mod2_fitcube[9, i, j].copy()
        snr_nii_2 = self.mod2_fitcube[10, i, j].copy()
        snr_ha_2 = self.mod2_fitcube[11, i, j].copy()
        vel_snr = self.mod2_fitcube[12, i, j]
        chi2 = self.mod2_fitcube[13, i, j]
        nii_ratio = popt2[0]/popt2[4] if popt2[0] < popt2[4] else popt2[4]/popt2[0]
        ha_ratio = popt2[1]/popt2[5] if popt2[1] < popt2[5] else popt2[5]/popt2[1]

        # Choose mod1 if well-fitted
        if snr_nii > 3. and snr_ha > 3. and chi1 <= 1.:
            self.mod2_fitcube[:,i,j] = np.nan
            self.mod2_fiterrcube[:,i,j] = np.nan
            return 1
        # Choose mod2 if the difference is significant
        if snr_nii_1 > 3. and snr_ha_1 > 3. and snr_nii_2 > 3. and snr_ha_2 > 3. and vel_snr > 3. and chi2 < chi1 and nii_ratio > 0.05 and ha_ratio > 0.05:

            self.mod1_fitcube[:,i,j] = np.nan
            self.mod1_fiterrcube[:,i,j] = np.nan 

            # Component left/right check
            if self.mod2_fitcube[2,i,j] > self.mod2_fitcube[6,i,j]:
                tmp = self.mod2_fitcube[0:4, i, j].copy()
                self.mod2_fitcube[0:4, i, j] = self.mod2_fitcube[4:8, i, j].copy()
                self.mod2_fitcube[4:8, i, j] = tmp.copy()
                tmp = self.mod2_fiterrcube[0:4, i, j].copy()
                self.mod2_fiterrcube[0:4, i, j] = self.mod2_fitcube[4:8, i, j].copy()
                self.mod2_fiterrcube[4:8, i, j] = tmp.copy()
                self.mod2_fitcube[8, i, j] = snr_nii_2
                self.mod2_fitcube[9, i, j] = snr_ha_2
                self.mod2_fitcube[10, i, j] = snr_nii_1
                self.mod2_fitcube[11, i, j] = snr_ha_1
               

            return 2
        elif snr_nii > 3. and snr_ha > 3.:
            self.mod2_fitcube[:,i,j] = np.nan
            self.mod2_fiterrcube[:,i,j] = np.nan
            return 1
        else:
            self.mod1_fitcube[:,i,j] = np.nan
            self.mod1_fiterrcube[:,i,j] = np.nan 
            self.mod2_fitcube[:,i,j] = np.nan
            self.mod2_fiterrcube[:,i,j] = np.nan
            return 0
        ...



    def component_sort(self, i, j):
        self.fitcube = np.zeros((14, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((8 ,self.cube_x, self.cube_y))

        snr_nii_1 = self.mod2_fitcube[8, i, j].copy()
        snr_ha_1 = self.mod2_fitcube[9, i, j].copy()
        snr_nii_2 = self.mod2_fitcube[10, i, j].copy()
        snr_ha_2 = self.mod2_fitcube[11, i, j].copy()

        # Component left/right check
        if self.mod2_fitcube[0,i,j] > self.mod2_fitcube[4,i,j]:
            tmp = self.mod2_fitcube[0:4, i, j].copy()
            self.mod2_fitcube[0:4, i, j] = self.mod2_fitcube[4:8, i, j].copy()
            self.mod2_fitcube[4:8, i, j] = tmp.copy()
            tmp = self.mod2_fiterrcube[0:4, i, j].copy()
            self.mod2_fiterrcube[0:4, i, j] = self.mod2_fitcube[4:8, i, j].copy()
            self.mod2_fiterrcube[4:8, i, j] = tmp.copy()
            
            self.mod2_fitcube[8, i, j] = snr_nii_2
            self.mod2_fitcube[9, i, j] = snr_ha_2
            self.mod2_fitcube[10, i, j] = snr_nii_1
            self.mod2_fitcube[11, i, j] = snr_ha_1
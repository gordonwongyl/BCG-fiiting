from turtle import color
from fit_routine import WLAX, Lines, lines, oiii_doublet, c, z, oiii_wratio
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class OIII(Lines):
    def __init__(self, name: str, rest, crange, vrange, cube: NDArray, varc: NDArray, contspec: NDArray):
        super().__init__(name, rest, crange, vrange, cube, varc, contspec)
        self.fitcube = np.zeros((5, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((4, self.cube_x, self.cube_y))

    def plot_spe(self, ax_in, i, j):
        self.spec = self.spaxel(self.subcube, i, j)
        self.varspec = self.spaxel(self.errcube, i, j)
        self.remove_nan()
        
        lranges = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])
        fit_spec = self.baseline_subtraction()
        

        popt = list(self.fitcube[0:4, i, j])
        fitsnr = self.fitcube[4, i, j]
        uncertainty = list(self.fiterrcube[:, i, j])
        
        quicksnr = sum(fit_spec)/np.sqrt(sum(self.varspec))

        base_cont_lvl = self.get_cont_lvl(self.basespec)
        line_cont_lvl = self.get_cont_lvl(self.spec)
        aC = (line_cont_lvl/base_cont_lvl) * self.basespec

        #plot
        ax_in[1].step(self.wlax, fit_spec, where='mid', color='black', label='Continuum range (Smask)')
        ax_in[1].step(self.wlax[lranges], fit_spec[lranges], where='mid', label = 'Line range')
        ax_in[1].plot(self.wlax[lranges], oiii_doublet(self.wlax[lranges], *popt), label='Line fitting')
        #ax_in[0].axhline(y=np.median(fit_spec[lranges]), color='r', label='Median around line ranges')
        ax_in[1].set_title(f"(i,j): ({j+1}, {i+1}), Residual Spectrum (R)")
        ax_in[1].legend(loc='upper right')
        ax_in[1].set(xlabel='Wavelength(Angstrom)', ylabel='F (10**-20 Angstrom-1 cm-2 ergs-1)')
        

        ax_in[0].step(self.wlax, self.spec, where='mid', color='black', label='Continuum range (Smask)')
        ax_in[0].step(self.wlax[lranges], self.spec[lranges], where='mid', label='Line range')
        ax_in[0].step(self.wlax, aC, where='mid', label='scaled template baseline (aC)')
        ax_in[0].axvspan(popt[1]-1.18*popt[2], popt[1]+1.18*popt[2], color='g', alpha=0.3)
        ax_in[0].set(xlabel='Wavelength(Angstrom)', ylabel='F (10**-20 Angstrom-1 cm-2 ergs-1)')
        ax_in[0].axvspan(oiii_wratio*(popt[1]-1.18*popt[2]), oiii_wratio*(popt[1]+1.18*popt[2]), color='g', alpha=0.3)  
        ax_in[0].legend(loc='upper right')
        ax_in[0].set_title(f"Original Spectrum (S) at ({j+1},{i+1}) scale factor (alpha) = {(line_cont_lvl/base_cont_lvl):.2f}")



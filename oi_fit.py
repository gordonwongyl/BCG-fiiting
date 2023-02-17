from fit_routine import WLAX, Lines, lines, oi_doublet,single, c, z, oi_wratio
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class OI(Lines):
    def __init__(self, name: str, rest, crange, vrange, cube: NDArray, varc: NDArray, contspec: NDArray):
        super().__init__(name, rest, crange, vrange, cube, varc, contspec)
        self.fitcube = np.zeros((6, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((4, self.cube_x, self.cube_y))

    def plot_spe(self, ax_in, i, j):
        self.spec = self.spaxel(self.subcube, i, j)
        self.varspec = self.spaxel(self.errcube, i, j)
        self.remove_nan()
        
        lranges = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])
        fit_spec = self.baseline_subtraction()
        

        popt = list(self.fitcube[0:4, i, j])
        fitsnr1 = self.fitcube[4, i, j]
        fitsnr2 = self.fitcube[5, i, j]
        uncertainty = list(self.fiterrcube[:, i, j])
        
        quicksnr = sum(fit_spec)/np.sqrt(sum(self.varspec))

        
        
        ratioerr = (popt[0]/popt[1])*np.sqrt((uncertainty[0]/popt[0])**2 + (uncertainty[1]/popt[1])**2)
        

        #plot
        ax_in[0].step(self.wlax, fit_spec, where='mid', color='black')
        ax_in[0].step(self.wlax[lranges], fit_spec[lranges], where='mid', color='#1f77b4')
        ax_in[0].axhline(y=np.median(fit_spec[~lranges]), color = 'green', label = 'Median of baseline')
        ax_in[0].plot(self.wlax, single(self.wlax, *popt), color='Orange')
        ax_in[0].axhline(y=np.median(fit_spec[lranges]), color='r', label='Median around line ranges')
        ax_in[0].set_title(f"(i,j): ({j+1}, {i+1}), quicksnr = {quicksnr}, fitsnr = {fitsnr1}")
        #ax_in[0].set_title(f"(i,j): ({j+1}, {i+1}), fitsnr = {fitsnr1}, {fitsnr2}, ratio = {popt[0]/popt[1]} rerr = {ratioerr} ")
        ax_in[0].legend(loc='upper right')

        ax_in[1].step(self.wlax, self.spec, where='mid')
        ax_in[1].axvspan(popt[1]-1.18*popt[2], popt[1]+1.18*popt[2], color='g', alpha=0.3)
        #ax_in[1].axvspan(popt[2]-1.18*popt[3], popt[2]+1.18*popt[3], color='g', alpha=0.3)
        #ax_in[1].axvspan(oi_wratio*(popt[2]-1.18*popt[3]), oi_wratio*(popt[2]+1.18*popt[3]), color='g', alpha=0.3)  

        ax_in[1].set_title(f"Original Spectrum at ({j+1},{i+1})")


   
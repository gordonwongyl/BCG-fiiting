from astropy.io import fits
import numpy as np
import scipy.interpolate as sip
from scipy.optimize import curve_fit
import sys
import copy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import interp1d

c = 2.9979e5
vsys = 2758.
gam = (1. + vsys/c)

def listvals(dict):
    return list(dict.values())

def listkeys(dict):
    return list(dict.keys())

#define the lines we're interested in:

def AIR(VAC):
    return VAC / (1.0 + 2.735182e-4 + 131.4182 / VAC**2 + 2.76249E8 / VAC**4)

lines = {'OIII4959': 4958.911, 'OIII5007': 5006.843, 'OI6300': AIR(6302.046), 'OI6363': AIR(6365.536), 'NII6548': 6548.05, 'Halpha': 6562.801, 'NII6584': 6583.45, 'Hbeta': 4861.363}

#make cutouts of regions of interest Ha, [OI] and [OIII]:

hdul = fits.open(sys.argv[1])
head = hdul[0].header
cubehdu = hdul[1]
cubehead = cubehdu.header
cube = cubehdu.data
varc = np.sqrt(hdul[2].data)
wlax = cubehead['CRVAL3'] + np.arange(cubehead['NAXIS3']) * cubehead['CD3_3']

lineranges = {'OIII': (4950., 5150.), 'OI': (6320., 6420.), 'Halpha': (6530., 6730.), 'Hbeta': (4850., 4950.)}
cubeslices = {
    'OI': cube[(lineranges['OI'][0] < wlax) & (wlax < lineranges['OI'][1])] * lines['OI6300'] / c,
    'Halpha': cube[(lineranges['Halpha'][0] < wlax) & (wlax < lineranges['Halpha'][1])] * lines['Halpha'] / c,
    'OIII': cube[(lineranges['OIII'][0] < wlax) & (wlax < lineranges['OIII'][1])] * lines['OIII5007'] / c,
    'Hbeta': cube[(lineranges['Hbeta'][0] < wlax) & (wlax < lineranges['Hbeta'][1])] * lines['Hbeta'] / c,
    }
errslices = {
    'OI': varc[(lineranges['OI'][0] < wlax) & (wlax < lineranges['OI'][1])] * lines['OI6300'] / c,
    'Halpha': varc[(lineranges['Halpha'][0] < wlax) & (wlax < lineranges['Halpha'][1])] * lines['Halpha'] / c,
    'OIII': varc[(lineranges['OIII'][0] < wlax) & (wlax < lineranges['OIII'][1])] * lines['OIII5007'] / c,
    'Hbeta': varc[(lineranges['Hbeta'][0] < wlax) & (wlax < lineranges['Hbeta'][1])] * lines['Hbeta'] / c
}
vaxes = {
    'OI': c * (wlax[(lineranges['OI'][0] < wlax) & (wlax < lineranges['OI'][1])] / lines['OI6300'] - 1.) - vsys,
    'Halpha': c * (wlax[(lineranges['Halpha'][0] < wlax) & (wlax < lineranges['Halpha'][1])] / lines['Halpha'] - 1.) - vsys,
    'OIII': c * (wlax[(lineranges['OIII'][0] < wlax) & (wlax < lineranges['OIII'][1])] / lines['OIII5007'] - 1.) - vsys,
    'Hbeta': c * (wlax[(lineranges['Hbeta'][0] < wlax) & (wlax < lineranges['Hbeta'][1])] / lines['Hbeta'] - 1.) - vsys
}

#extract an off-nebula baseline spectrum from the data cube to be subtracted from the on-nebla specs:

basespecs = {}

basemask = fits.open('../v12/n5044_offneb_3sigmask.fits')[0].data

for str in ['Halpha', 'OI', 'OIII']:
    basespecs[str] = np.array([np.ma.median(np.ma.masked_array(arr, mask = basemask)) for arr in cubeslices[str]])

#interpolate over the telluric [OI] feature in the baseline:

badidx = (c * (6360. / lines['OI6300'] - 1.) - vsys < vaxes['OI']) & (vaxes['OI'] < c * (6368. / lines['OI6300'] - 1.) - vsys)

fig, axs = plt.subplots(3,1, figsize = (4,6))
# fig.subplots_adjust(hspace=0)
# axs[0].set_ylim(np.min(grflux_vs[0][i]), np.min(grflux_vs[0][i]) + 1.05*np.ptp(grflux_vs[0][i]))
axs[0].plot(wlax[(lineranges['Halpha'][0] < wlax) & (wlax < lineranges['Halpha'][1])], basespecs['Halpha'], 'k-', label=r'H$\alpha$+[NII] baseline')
axs[0].legend()
# axs[1].set_ylim(np.min(grflux_vs[1][i]), np.min(grflux_vs[1][i]) + 1.05*np.ptp(grflux_vs[1][i]))
axs[1].plot(wlax[(lineranges['OI'][0] < wlax) & (wlax < lineranges['OI'][1])][~badidx], basespecs['OI'][~badidx], 'k-', label=r'[OI] baseline')
axs[1].plot(wlax[(lineranges['OI'][0] < wlax) & (wlax < lineranges['OI'][1])][badidx], basespecs['OI'][badidx], 'k--', alpha = 0.3, label=r'masked')
axs[1].legend()
axs[2].set_ylabel(r'\si{erg.s^{-1}.cm^{-2}.(km.s^{-1})^{-1}')
axs[2].set_xlabel(r'\si{\AA}')
# axs[2].set_ylim(np.min(grflux_vs[2][i-9]), np.min(grflux_vs[2][i-9]) + 1.05*np.ptp(grflux_vs[2][i-9]))
axs[2].plot(wlax[(lineranges['OIII'][0] < wlax) & (wlax < lineranges['OIII'][1])], basespecs['OIII'], 'k-', label=r'[OIII] baseline')
axs[2].legend()
plt.savefig('./testspecs/baselines.pdf', bbox_inches='tight')
# plt.savefig('./testspecs/ha_h2_co_finegrid_wmods_v02_%02d.png' %i, bbox_inches = 'tight', overwrite=True)
plt.clf()

baseinterp = interp1d(vaxes['OI'][~badidx], basespecs['OI'][~badidx], kind='cubic')
basespecs['OI'] = baseinterp(vaxes['OI'])

# fig, axs = plt.subplots(3,1, sharex=True, figsize = (4,6))
# fig.subplots_adjust(hspace=0)
# # axs[0].set_ylim(np.min(grflux_vs[0][i]), np.min(grflux_vs[0][i]) + 1.05*np.ptp(grflux_vs[0][i]))
# axs[0].plot(wlax[(lineranges['Halpha'][0] < wlax) & (wlax < lineranges['Halpha'][1])], basespecs['Halpha'], 'k-')
# axs[0].legend()
# # axs[1].set_ylim(np.min(grflux_vs[1][i]), np.min(grflux_vs[1][i]) + 1.05*np.ptp(grflux_vs[1][i]))
# axs[1].plot(wlax[(lineranges['OI'][0] < wlax) & (wlax < lineranges['OI'][1])], basespecs['OI'], 'k-')
# axs[1].legend()
# axs[2].set_ylabel(r'\si{erg.s^{-1}.cm^{-2}.(km.s^{-1})^{-1}.arcsec^{-2}')
# axs[2].set_xlabel(r'\si{\AA}')
# # axs[2].set_ylim(np.min(grflux_vs[2][i-9]), np.min(grflux_vs[2][i-9]) + 1.05*np.ptp(grflux_vs[2][i-9]))
# axs[2].plot(wlax[(lineranges['OIII'][0] < wlax) & (wlax < lineranges['OIII'][1])], basespecs['OIII'], 'k-')
# axs[2].legend()
# plt.savefig('./testspecs/baselines.pdf', bbox_inches='tight', overwrite=True)
# # plt.savefig('./testspecs/ha_h2_co_finegrid_wmods_v02_%02d.png' %i, bbox_inches = 'tight', overwrite=True)
# plt.clf()

# perform fitting on each spaxel of each cube:
## define the models:

deltniil = c * (lines['NII6584'] - lines['Halpha']) / lines['Halpha']
deltniis = c * (lines['NII6548'] - lines['Halpha']) / lines['Halpha']
deltois = c * (lines['OI6363'] - lines['OI6300']) / lines['OI6300']
deltoiiis = c * (lines['OIII4959'] - lines['OIII5007']) / lines['OIII5007']

def hamod1(x, h1, v01, dv1, n1):
    return (h1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-v01)**2)/(2*dv1**2)) + (n1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-(v01 + deltniil))**2)/(2*dv1**2)) + ((n1/np.sqrt(2*np.pi)/dv1)/3.)*np.exp(-((x-(v01 + deltniis))**2)/(2*dv1**2))
def hamod2(x, h1, v01, dv1, n1, h2, v02, dv2, n2):
    return (h1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-v01)**2)/(2*dv1**2)) + (n1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-(v01 + deltniil))**2)/(2*dv1**2)) + ((n1/np.sqrt(2*np.pi)/dv1)/3.)*np.exp(-((x-(v01 + deltniis))**2)/(2*dv1**2)) + (h2/np.sqrt(2*np.pi)/dv2)*np.exp(-((x-v02)**2)/(2*dv2**2)) + (n2/np.sqrt(2*np.pi)/dv2)*np.exp(-((x-(v02 + deltniil))**2)/(2*dv2**2)) + ((n2/np.sqrt(2*np.pi)/dv2)/3.)*np.exp(-((x-(v02 + deltniis))**2)/(2*dv2**2))

# correct any Ha absorption in the MUSE specs by adding a gaussian with the following parameters:

absew = 80.5
absdv = 366.
absv0 = 14.6

def contlvl(x, c):
    return c

def spaxel(arr, i, j):
    return np.array([x[i, j] for x in arr])

hacube = np.zeros((8, np.shape(cubeslices['Halpha'])[1], np.shape(cubeslices['Halpha'])[2]))
hacuberr = np.zeros((8, np.shape(cubeslices['Halpha'])[1], np.shape(cubeslices['Halpha'])[2]))
oicube = np.zeros((2, np.shape(cubeslices['Halpha'])[1], np.shape(cubeslices['Halpha'])[2]))
oicuberr = np.zeros((2, np.shape(cubeslices['Halpha'])[1], np.shape(cubeslices['Halpha'])[2]))
oiiicube = np.zeros((2, np.shape(cubeslices['Halpha'])[1], np.shape(cubeslices['Halpha'])[2]))
oiiicuberr = np.zeros((2, np.shape(cubeslices['Halpha'])[1], np.shape(cubeslices['Halpha'])[2]))

pixels = [[53,58], [63,95], [80,95], [57,42], [57,15], [57,57], [80,97], [84,100], [76,94], [59,8], [70,43], [32,57], [66,31], [58,14], [44,80], [55,16], [42,32], [44,49], [49,44], [57,54], [46,66], [66,46], [40,57],[57,40]]

#define apertures for taking global specs from inner regions:
from regions import read_ds9, PixCoord

apers = read_ds9('globalspecapers.reg')
haglobspec = [np.zeros(np.size(spaxel(cubeslices['Halpha'], 0, 0))) for n in range(len(apers)-1)]
haglobvar = [np.zeros(np.size(spaxel(errslices['Halpha']**2., 0, 0))) for n in range(len(apers)-1)]
hacounter = np.zeros(len(apers)-1)
oiglobspec = [np.zeros(np.size(spaxel(cubeslices['OI'], 0, 0))) for n in range(len(apers)-1)]
oiglobvar = [np.zeros(np.size(spaxel(errslices['OI']**2., 0, 0))) for n in range(len(apers)-1)]
oicounter = np.zeros(len(apers)-1)
oiiiglobspec = [np.zeros(np.size(spaxel(cubeslices['OIII'], 0, 0))) for n in range(len(apers)-1)]
oiiiglobvar = [np.zeros(np.size(spaxel(errslices['OIII']**2., 0, 0))) for n in range(len(apers)-1)]
oiiicounter = np.zeros(len(apers)-1)

stime = time.time()

print('\nLine emission image extraction started...\n')

for i in tqdm(range(np.shape(cubeslices['Halpha'])[1])):
    for j in range(np.shape(cubeslices['Halpha'])[2]):
        haspec = copy.deepcopy(spaxel(cubeslices['Halpha'], i, j))
        oispec = copy.deepcopy(spaxel(cubeslices['OI'], i, j))
        oiiispec = copy.deepcopy(spaxel(cubeslices['OIII'], i, j))
        haerr = copy.deepcopy(spaxel(errslices['Halpha'], i, j))
        oierr = copy.deepcopy(spaxel(errslices['OI'], i, j))
        oiiierr = copy.deepcopy(spaxel(errslices['OIII'], i, j))
        habasespec = copy.deepcopy(basespecs['Halpha'])
        oibasespec = copy.deepcopy(basespecs['OI'])
        oiiibasespec = copy.deepcopy(basespecs['OIII'])
        haspec[np.isnan(haspec)] = 50.
        oispec[np.isnan(oispec)] = 50.
        oiiispec[np.isnan(oiiispec)] = 50.
        haerr[np.isnan(haerr)] = 0.
        oierr[np.isnan(oierr)] = 0.
        oiiierr[np.isnan(oiiierr)] = 0.
        # haclvl = curve_fit(contlvl, vaxes['Halpha'], haspec, p0 = [1000])
        haidx = (-1000 < vaxes['Halpha']) & (vaxes['Halpha'] < 1300)
        # clvl, med, std = sigma_clipped_stats(haspec[~idx], maxiters = 10)
        # bclvl, med, std = sigma_clipped_stats(habasespec[~idx], maxiters = 10)
        clvl = np.average(haspec[~haidx])
        bclvl = np.average(habasespec[~haidx])
        oiidx = (-500 < vaxes['OI']) & (vaxes['OI'] < 500)
        # oiclvl, med, std = sigma_clipped_stats(oispec[~idx], maxiters = 10)
        # oibclvl, med, std = sigma_clipped_stats(oibasespec[~idx], maxiters = 10)
        oiclvl = np.average(oispec[~oiidx])
        oibclvl = np.average(oibasespec[~oiidx])
        oiiiidx = (-500 < vaxes['OIII']) & (vaxes['OIII'] < 500)
        # oiiiclvl, med, std = sigma_clipped_stats(oiiispec[~idx], maxiters = 10)
        # oiiibclvl, med, std = sigma_clipped_stats(oiiibasespec[~idx], maxiters = 10)
        oiiiclvl = np.average(oiiispec[~oiiiidx])
        oiiibclvl = np.average(oiiibasespec[~oiiiidx])
        # def cormod(x):
        #     return (clvl * absew / np.sqrt(2 * np.pi) / absdv) * np.exp(-(x - absv0) ** 2 / 2 / absdv ** 2)
        # haspec += cormod(vaxes['Halpha'])
        haspec -= (clvl / bclvl) * habasespec
        oispec -= (oiclvl / oibclvl) * oibasespec
        oiiispec -= (oiiiclvl / oiiibclvl) * oiiibasespec
        oifr = (-2000 < vaxes['OI']) & (vaxes['OI'] < 2000)
        oiiifr = (-2000 < vaxes['OIII']) & (vaxes['OIII'] < 2000)
        def hamod1(x, h1, v01, dv1, n1):
            return (h1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-v01)**2)/(2*dv1**2)) + (n1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-(v01 + deltniil))**2)/(2*dv1**2)) + ((n1/np.sqrt(2*np.pi)/dv1)/3.)*np.exp(-((x-(v01 + deltniis))**2)/(2*dv1**2))
        def hamod2(x, h1, v01, dv1, n1, h2, v02, dv2, n2):
            return (h1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-v01)**2)/(2*dv1**2)) + (n1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-(v01 + deltniil))**2)/(2*dv1**2)) + ((n1/np.sqrt(2*np.pi)/dv1)/3.)*np.exp(-((x-(v01 + deltniis))**2)/(2*dv1**2)) + (h2/np.sqrt(2*np.pi)/dv2)*np.exp(-((x-v02)**2)/(2*dv2**2)) + (n2/np.sqrt(2*np.pi)/dv2)*np.exp(-((x-(v02 + deltniil))**2)/(2*dv2**2)) + ((n2/np.sqrt(2*np.pi)/dv2)/3.)*np.exp(-((x-(v02 + deltniis))**2)/(2*dv2**2))

        try:
            popt1, pcov1 = curve_fit(hamod1, vaxes['Halpha'], haspec, p0 = [40000, 0, 60, 75000], bounds = ([-3e6, -300., 50., -8e6], [1e10, 300., 600., 8e10]), absolute_sigma=True, sigma=haerr)
            chisq1 = np.sum((haspec - hamod1(vaxes['Halpha'], *popt1))**2. / haerr**2.) / (np.size(haspec) - np.size(popt1))
            if chisq1 <= 1.0:
                hacube[:4,i,j] = np.nan
                hacuberr[:4,i,j] = np.nan
                hacube[4:,i, j] = popt1[0:4]
                hacuberr[4:, i, j] = np.diagonal(np.sqrt(np.abs(pcov1))[0:4])
                popt = popt1
            else:
                def hamod2(x, h1, v01, dv1, n1, h2, v02, dv2, n2):
                    ...
                popt2, pcov2 = curve_fit(hamod2, vaxes['Halpha'], haspec, p0 = [40000, -100, 60, 75000,40000, 0, 60, 75000], bounds = ([-3e6, -400., 50., -8e6,-3e6, -500., 50., -8e6], [1e10, 400., 600., 8e10,1e10, 400., 600., 8e10]), absolute_sigma=True, sigma=haerr)
                chisq2 = np.sum((haspec - hamod2(vaxes['Halpha'], *popt2))**2. / haerr**2.) / (np.size(haspec) - np.size(popt2))
                if ((popt2[0] > 3 * np.sqrt(pcov2[0,0])) & (popt2[4] > 3 * np.sqrt(pcov2[4,4])) & (np.abs(popt2[1] - popt2[5]) / np.sqrt(pcov2[1,1] + pcov2[5,5]) > 3.)):
                    hacube[:,i, j] = popt2[0:8]
                    hacuberr[:, i, j] = np.diagonal(np.sqrt(np.abs(pcov2))[0:8])
                    popt = popt2
                else:
                    hacube[:4,i,j] = np.nan
                    hacuberr[:4,i,j] = np.nan
                    hacube[4:,i, j] = popt1[0:4]
                    hacuberr[4:, i, j] = np.diagonal(np.sqrt(np.abs(pcov1))[0:4])
                    popt = popt1
        except RuntimeError:
            hacube[:8, i, j] = np.nan
            hacuberr[:8, i, j] = np.nan
        if len(popt) == 8:
            def hamod1(x, h1, v01, dv1, n1, h2, v02, dv2, n2):
                return (h1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-v01)**2)/(2*dv1**2)) + (n1/np.sqrt(2*np.pi)/dv1)*np.exp(-((x-(v01 + deltniil))**2)/(2*dv1**2)) + ((n1/np.sqrt(2*np.pi)/dv1)/3.)*np.exp(-((x-(v01 + deltniis))**2)/(2*dv1**2)) + (h2/np.sqrt(2*np.pi)/dv2)*np.exp(-((x-v02)**2)/(2*dv2**2)) + (n2/np.sqrt(2*np.pi)/dv2)*np.exp(-((x-(v02 + deltniil))**2)/(2*dv2**2)) + ((n2/np.sqrt(2*np.pi)/dv2)/3.)*np.exp(-((x-(v02 + deltniis))**2)/(2*dv2**2))
            def oimod1(x, oi1, oi2):
                return (oi1/np.sqrt(2*np.pi)/np.abs(hacube[2,i,j]))*np.exp(-((x-hacube[1,i,j])**2)/(2*hacube[2,i,j]**2)) + (oi2/np.sqrt(2*np.pi)/np.abs(hacube[6,i,j]))*np.exp(-((x-hacube[5,i,j])**2)/(2*hacube[6,i,j]**2))
            def oiiimod1(x, oiii1, oiii2):
                return (oiii1/np.sqrt(2*np.pi)/np.abs(hacube[2,i,j]))*np.exp(-((x-hacube[1,i,j])**2)/(2*hacube[2,i,j]**2)) \
                + (oiii2/np.sqrt(2*np.pi)/np.abs(hacube[6,i,j]))*np.exp(-((x-hacube[5,i,j])**2)/(2*hacube[6,i,j]**2))
            try:
                poptoi, pcovoi = curve_fit(oimod1, vaxes['OI'][oifr], oispec[oifr], p0 = [4000, 4000], bounds = ([-1e9, -1e9], [1e9, 1e9]), absolute_sigma=True, sigma=oierr[oifr])
                oicube[:,i, j] = poptoi
                oicuberr[:,i, j] = np.sqrt(np.average(np.square(oispec[~oiidx]))) * np.abs(popt2[[2,6]]) * np.sqrt(2*np.pi)
            except (RuntimeError, ValueError):
                oicube[:,i, j] = np.nan
                oicuberr[:,i, j] = np.nan
                poptoi = [np.nan,np.nan]
            try:
                poptoiii, pcovoiii = curve_fit(oiiimod1, vaxes['OIII'][oiiifr], oiiispec[oiiifr], p0 = [4000, 4000], bounds = ([-1e9, -1e9], [1e9, 1e9]), absolute_sigma=True, sigma=oiiierr[oiiifr])
                oiiicube[:,i, j] = poptoiii
                oiiicuberr[:,i, j] = np.sqrt(np.average(np.square(oiiispec[~oiiiidx]))) * np.abs(popt2[[2,6]]) * np.sqrt(2*np.pi)
            except (RuntimeError, ValueError):
                oiiicube[:,i, j] = np.nan
                oiiicuberr[:,i, j] = np.nan
                poptoiii = [np.nan,np.nan]
        if len(popt) == 4:
            def oimod1(x, oi1):
                return (oi1/np.sqrt(2*np.pi)/np.abs(hacube[6,i,j]))*np.exp(-((x-hacube[5,i,j])**2)/(2*hacube[6,i,j]**2))
            def oiiimod1(x, oiii1):
                return (oiii1/np.sqrt(2*np.pi)/np.abs(hacube[6,i,j]))*np.exp(-((x-hacube[5,i,j])**2)/(2*hacube[6,i,j]**2))
            try:
                poptoi, pcovoi = curve_fit(oimod1, vaxes['OI'][oifr], oispec[oifr], p0 = [40000], bounds = ([-1e9], [1e9]), absolute_sigma=True, sigma=oierr[oifr])
                oicube[1:,i, j] = poptoi[0]
                oicuberr[1:,i, j] = np.sqrt(np.average(np.square(oispec[~oiidx]))) * np.abs(popt[2]) * np.sqrt(2*np.pi)
            except (RuntimeError, ValueError):
                oicube[:,i, j] = np.nan
                oicuberr[:,i, j] = np.nan
                poptoi = [np.nan]
            try:
                poptoiii, pcovoiii = curve_fit(oiiimod1, vaxes['OIII'][oiiifr], oiiispec[oiiifr], p0 = [40000], bounds = ([-1e9], [1e9]), absolute_sigma=True, sigma=oiiierr[oiiifr])
                oiiicube[1:,i, j] = poptoiii[0]
                oiiicuberr[1:,i, j] = np.sqrt(np.average(np.square(oiiispec[~oiiiidx]))) * np.abs(popt[2]) * np.sqrt(2*np.pi)
            except (RuntimeError, ValueError):
                oiiicube[:,i, j] = np.nan
                oiiicuberr[:,i, j] = np.nan
                poptoiii = [np.nan]
        for pix in pixels:
            if ((i == pix[1]) & (j == pix[0])):
                haidx = (vaxes['Halpha'] > -2000) & (vaxes['Halpha'] < 2000)
                oiidx = (vaxes['OI'] > -2000) & (vaxes['OI'] < 2000)
                oiiiidx = (vaxes['OIII'] > -2000) & (vaxes['OIII'] < 2000)
                finevel = np.linspace(-2000.,2000.,1000)
                # fig, axs = plt.subplots(3,1, sharex=True, figsize = (4,6))
                # fig.subplots_adjust(hspace=0)
                # # axs[0].set_ylim(np.min(grflux_vs[0][i]), np.min(grflux_vs[0][i]) + 1.05*np.ptp(grflux_vs[0][i]))
                # axs[0].errorbar(vaxes['Halpha'][haidx], haspec[haidx], yerr = haerr[haidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'H$\alpha$ + [NII]')
                # # axs[0].plot(finevel, hamod1(finevel, *popt), 'r--', label = r'Model', alpha = 0.3)
                # axs[0].legend()
                # # axs[1].set_ylim(np.min(grflux_vs[1][i]), np.min(grflux_vs[1][i]) + 1.05*np.ptp(grflux_vs[1][i]))
                # axs[1].errorbar(vaxes['OI'][oiidx], oispec[oiidx], yerr = oierr[oiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'[OI]')
                # # axs[1].plot(finevel, oimod1(finevel, *poptoi), 'r--', label = r'Model', alpha = 0.3)
                # axs[1].legend()
                # axs[2].set_ylabel(r'$10^{-20}$ \si{erg.s^{-1}.cm^{-2}.(km.s^{-1})^{-1}}')
                # axs[2].set_xlabel(r'\si{km.s^{-1}}')
                # # axs[2].set_ylim(np.min(grflux_vs[2][i-9]), np.min(grflux_vs[2][i-9]) + 1.05*np.ptp(grflux_vs[2][i-9]))
                # axs[2].errorbar(vaxes['OIII'][oiiiidx], oiiispec[oiiiidx], yerr = oiiierr[oiiiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'[OIII]')
                # # axs[2].plot(finevel, oiiimod1(finevel, *poptoiii), 'r--', label = r'Model', alpha = 0.3)
                # axs[2].legend()
                # plt.savefig('./testspecs/spaxel_%s_%s.pdf' %(pix[0],pix[1]), bbox_inches='tight')
                # # plt.savefig('./testspecs/ha_h2_co_finegrid_wmods_v02_%02d.png' %i, bbox_inches = 'tight', overwrite=True)
                # plt.clf()
                fig, axs = plt.subplots(3,1, sharex=True, figsize = (4,6))
                fig.subplots_adjust(hspace=0)
                # axs[0].set_ylim(np.min(grflux_vs[0][i]), np.min(grflux_vs[0][i]) + 1.05*np.ptp(grflux_vs[0][i]))
                axs[0].errorbar(vaxes['Halpha'][haidx], haspec[haidx], yerr = haerr[haidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'H$\alpha$ + [NII]')
                axs[0].plot(finevel, hamod1(finevel, *popt), 'r--', label = r'Model', alpha = 0.3)
                axs[0].legend()
                # axs[1].set_ylim(np.min(grflux_vs[1][i]), np.min(grflux_vs[1][i]) + 1.05*np.ptp(grflux_vs[1][i]))
                axs[1].errorbar(vaxes['OI'][oiidx], oispec[oiidx], yerr = oierr[oiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'[OI]')
                axs[1].plot(finevel, oimod1(finevel, *poptoi), 'r--', label = r'Model', alpha = 0.3)
                axs[1].legend()
                axs[2].set_ylabel(r'$10^{-20}$ \si{erg.s^{-1}.cm^{-2}.(km.s^{-1})^{-1}}')
                axs[2].set_xlabel(r'\si{km.s^{-1}}')
                # axs[2].set_ylim(np.min(grflux_vs[2][i-9]), np.min(grflux_vs[2][i-9]) + 1.05*np.ptp(grflux_vs[2][i-9]))
                axs[2].errorbar(vaxes['OIII'][oiiiidx], oiiispec[oiiiidx], yerr = oiiierr[oiiiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'[OIII]')
                axs[2].plot(finevel, oiiimod1(finevel, *poptoiii), 'r--', label = r'Model', alpha = 0.3)
                axs[2].legend()
                plt.savefig('./testspecs/2peak_spaxel_wmod_%s_%s.pdf' %(pix[0],pix[1]), bbox_inches='tight')
                # plt.savefig('./testspecs/ha_h2_co_finegrid_wmods_v02_%02d.png' %i, bbox_inches = 'tight', overwrite=True)
                plt.clf()
        # for k in range(1,len(apers)):
        #     if (~(PixCoord(j,i) in apers[0]) & (PixCoord(j,i) in apers[k]) & (oiiicube[i,j] > 3 * oiiicuberr[i,j]) & ((oicube[i,j] > 3 * oicuberr[i,j]))):
        #         haglobspec[k-1] += haspec
        #         haglobvar[k-1] += haerr**2.
        #         hacounter[k-1] += 1.
        #         oiglobspec[k-1] += oispec
        #         oiglobvar[k-1] += oierr**2.
        #         oicounter[k-1] += 1.
        #         oiiiglobspec[k-1] += oiiispec
        #         oiiiglobvar[k-1] += oiiierr**2.
        #         oiiicounter[k-1] += 1.

# hastrongoiiispec = (haglobspec[0] + haglobspec[1]) / (hacounter[0] + hacounter[1])
# hastrongoiiierr = np.sqrt(haglobvar[0] + haglobvar[1]) / (hacounter[0] + hacounter[1])
# haweakoiiispec = (haglobspec[2] + haglobspec[3] + haglobspec[4]) / (hacounter[2] + hacounter[3] + hacounter[4])
# haweakoiiierr = np.sqrt(haglobvar[2] + haglobvar[3] + haglobvar[4]) / (hacounter[2] + hacounter[3] + hacounter[4])
# oistrongoiiispec = (oiglobspec[0] + oiglobspec[1]) / (oicounter[0] + oicounter[1])
# oistrongoiiierr = np.sqrt(oiglobvar[0] + oiglobvar[1]) / (oicounter[0] + oicounter[1])
# oiweakoiiispec = (oiglobspec[2] + oiglobspec[3] + oiglobspec[4]) / (oicounter[2] + oicounter[3] + oicounter[4])
# oiweakoiiierr = np.sqrt(oiglobvar[2] + oiglobvar[3] + oiglobvar[4]) / (oicounter[2] + oicounter[3] + oicounter[4])
# oiiistrongoiiispec = (oiiiglobspec[0] + oiiiglobspec[1]) / (oiiicounter[0] + oiiicounter[1])
# oiiistrongoiiierr = np.sqrt(oiiiglobvar[0] + oiiiglobvar[1]) / (oiiicounter[0] + oiiicounter[1])
# oiiiweakoiiispec = (oiiiglobspec[2] + oiiiglobspec[3] + oiiiglobspec[4]) / (oiiicounter[2] + oiiicounter[3] + oiiicounter[4])
# oiiiweakoiiierr = np.sqrt(oiiiglobvar[2] + oiiiglobvar[3] + oiiiglobvar[4]) / (oiiicounter[2] + oiiicounter[3] + oiiicounter[4])
#
# fig, axs = plt.subplots(3,1, sharex=True, figsize = (4,6))
# fig.subplots_adjust(hspace=0)
# haidx = (vaxes['Halpha'] > -2000) & (vaxes['Halpha'] < 2000)
# oiidx = (vaxes['OI'] > -2000) & (vaxes['OI'] < 2000)
# oiiiidx = (vaxes['OIII'] > -2000) & (vaxes['OIII'] < 2000)
# axs[0].errorbar(vaxes['Halpha'][haidx], hastrongoiiispec[haidx], yerr=hastrongoiiierr[haidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'H$\alpha$ + [NII]')
# axs[0].legend()
# axs[1].errorbar(vaxes['OI'][oiidx], oistrongoiiispec[oiidx], yerr=oistrongoiiierr[oiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label='[OI]')
# axs[1].set_ylim([-3,10])
# axs[1].legend()
# axs[2].errorbar(vaxes['OIII'][oiiiidx], oiiistrongoiiispec[oiiiidx], yerr=oiiistrongoiiierr[oiiiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label='[OIII]')
# axs[2].set_ylim([-3,10])
# axs[2].legend()
# axs[2].set_xlabel(r'\si{km.s^{-1}}')
# axs[2].set_ylabel(r'$10^{-20}$ \si{erg.s^{-1}.cm^{-2}.(km.s^{-1})^{-1}}')
# plt.savefig('./testspecs/globavg_strongoiii_spec.pdf')
# plt.clf()
#
# fig, axs = plt.subplots(3,1, sharex=True, figsize = (4,6))
# fig.subplots_adjust(hspace=0)
# haidx = (vaxes['Halpha'] > -2000) & (vaxes['Halpha'] < 2000)
# oiidx = (vaxes['OI'] > -2000) & (vaxes['OI'] < 2000)
# oiiiidx = (vaxes['OIII'] > -2000) & (vaxes['OIII'] < 2000)
# axs[0].errorbar(vaxes['Halpha'][haidx], haweakoiiispec[haidx], yerr=haweakoiiierr[haidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label=r'H$\alpha$ + [NII]')
# axs[0].legend()
# axs[1].errorbar(vaxes['OI'][oiidx], oiweakoiiispec[oiidx], yerr=oiweakoiiierr[oiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label='[OI]')
# axs[1].set_ylim([-2,4])
# axs[1].legend()
# axs[2].errorbar(vaxes['OIII'][oiiiidx], oiiiweakoiiispec[oiiiidx], yerr=oiiiweakoiiierr[oiiiidx], elinewidth = .3, capthick = 0.3, linestyle = '', color = 'black', marker = '.', label='[OIII]')
# axs[2].set_ylim([-2,4])
# axs[2].legend()
# axs[2].set_xlabel(r'\si{km.s^{-1}}')
# axs[2].set_ylabel(r'$10^{-20}$ \si{erg.s^{-1}.cm^{-2}.(km.s^{-1})^{-1}}')
# plt.savefig('./testspecs/globavg_weakoiii_spec.pdf')
# plt.clf()

from astropy.wcs import WCS

newwcs = WCS(cubehead, naxis=2)
newhead = newwcs.to_header()
prihdu = fits.PrimaryHDU(hacube[0], header=newhead)
newhahdus = [fits.ImageHDU(hacube[i]) for i in range(1,8)]
haerrhdus = [fits.ImageHDU(hacuberr[i]) for i in range(8)]
hahdul = fits.HDUList([prihdu] + newhahdus + haerrhdus)
hahdul.writeto('n5044_muse_ha_2peak_fitmap_v' + sys.argv[0][-5:-3] + '.fits', overwrite = True)
prioihdu = fits.PrimaryHDU(oicube[0], header=newhead)
newoihdus = [fits.ImageHDU(oicube[1])]
oierrhdus = [fits.ImageHDU(oicuberr[i]) for i in range(2)]
oihdul = fits.HDUList([prioihdu] + newoihdus + oierrhdus)
oihdul.writeto('n5044_muse_oi_2peak_fitmap_v' + sys.argv[0][-5:-3] + '.fits', overwrite = True)
prioiiihdu = fits.PrimaryHDU(oiiicube[0], header=newhead)
newoiiihdus = [fits.ImageHDU(oiiicube[1])]
oiiierrhdus = [fits.ImageHDU(oiiicuberr[i]) for i in range(2)]
oiiihdul = fits.HDUList([prioiiihdu] + newoiiihdus + oiiierrhdus)
oiiihdul.writeto('n5044_muse_oiii_2peak_fitmap_v' + sys.argv[0][-5:-3] + '.fits', overwrite = True)
fits.PrimaryHDU(np.divide(oicube,oicuberr)).writeto('oisigcheck.fits',overwrite=True)

print('\nProcessing finished after %s seconds\n' %(time.time() - stime))

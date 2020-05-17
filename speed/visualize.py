import matplotlib.pyplot as plt
import h5py
from brutus import filters
import numpy as np

filt = filters.ps[:-2] + filters.tmass

from brutus.utils import load_offsets, load_models
(models_mist, labels_mist,
 lmask_mist) = load_models('../data/DATAFILES/grid_mist_v8.h5',
                                  filters=filt, include_ms=True,
                                  include_postms=True, include_binaries=True)

zp_mist = load_offsets('../data/DATAFILES/offsets_mist_v8.txt',
                                 filters=filt)



from brutus.utils import inv_magnitude
filename = 'Orion_l204.7_b-19.2'
f = h5py.File(filename+'.h5')
fpix = f['photometry']['pixel 0-0']
mag, magerr = fpix['mag'], fpix['err']
mask = np.isfinite(magerr)  # create boolean band mask
phot, err = inv_magnitude(mag, magerr)  # convert to flux
objid = fpix['obj_id']
parallax, parallax_err = fpix['parallax'] * 1e3, fpix['parallax_error'] * 1e3  # convert to mas
psel = np.isclose(parallax_err, 0.) | np.isclose(parallax, 0.) | (parallax_err > 1e6)
parallax[psel], parallax_err[psel] = np.nan, np.nan
coords = np.c_[fpix['l'], fpix['b']]





f = h5py.File(filename+'_mist'+'.h5', mode='r')
idxs_mist = f['model_idx'][:]
chi2_mist = f['obj_chi2min'][:]
nbands_mist = f['obj_Nbands'][:]
dists_mist = f['samps_dist'][:]
reds_mist = f['samps_red'][:]
dreds_mist = f['samps_dred'][:]


from brutus import plotting as bplot

# pick an object
i = 3

# MIST
fig, ax, parts = bplot.posterior_predictive(models_mist, idxs_mist[i],
                                            reds_mist[i], dreds_mist[i], dists_mist[i],
                                            data=phot[i], data_err=err[i],
                                            data_mask=mask[i],
                                            offset=zp_mist, psig=2.,
                                            labels=filt, vcolor='blue', pcolor='black')
plt.title('MIST')
plt.tight_layout()
plt.savefig('vis/vis_Orion_MIST.pdf')


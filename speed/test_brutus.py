import brutus
from brutus import fitting, filters
from brutus.utils import inv_magnitude, load_models, load_offsets

import h5py
import numpy as np

import os
if os.path.exists('Orion_l204.7_b-19.2_mist.h5'):
    os.remove('Orion_l204.7_b-19.2_mist.h5')

filt = filters.ps[:-2] + filters.tmass

# load in models
(models_mist, labels_mist,
 lmask_mist) = load_models('../data/DATAFILES/grid_mist_v8.h5', 
                                  filters=filt, include_ms=True, 
                                  include_postms=True, include_binaries=True)

# load in offsets
zp_mist = load_offsets('../data/DATAFILES/offsets_mist_v8.txt', 
                                 filters=filt)

# load in fitter
BF_mist = fitting.BruteForce(models_mist, labels_mist, lmask_mist)

# load in data
filename = 'Orion_l204.7_b-19.2'
f = h5py.File(filename+'.h5', mode='r')
fpix = f['photometry']['pixel 0-0']
mag, magerr = fpix['mag'], fpix['err']
mask = np.isfinite(magerr)  # create boolean band mask
phot, err = inv_magnitude(mag, magerr)  # convert to flux
objid = fpix['obj_id']
parallax, parallax_err = fpix['parallax'] * 1e3, fpix['parallax_error'] * 1e3  # convert to mas
psel = np.isclose(parallax_err, 0.) | np.isclose(parallax, 0.) | (parallax_err > 1e6)
parallax[psel], parallax_err[psel] = np.nan, np.nan
coords = np.c_[fpix['l'], fpix['b']]

BF_mist.fit(phot[:10], err[:10], mask[:10], objid[:10], 
            filename+'_mist',
            parallax=parallax[:10], parallax_err=parallax_err[:10], 
            data_coords=coords[:10], 
            dustfile='../data/DATAFILES/bayestar2019_v1.h5',
            phot_offsets=zp_mist,
            running_io=True,
            save_dar_draws=True,
            verbose=True)


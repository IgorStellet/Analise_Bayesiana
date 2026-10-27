import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import camb, os, h5py
from astropy.table import Table
from astropy.io import fits
from scipy.integrate import trapezoid, quad

c = 299792.458 ; ns=0.96

def load_SN_data():
    dtype = {
    'names': ('CID', 'IDSURVEY', 'zHD', 'zHDERR', 'zCMB', 'zCMBERR', 'zHEL',
              'zHELERR', 'm_b_corr', 'm_b_corr_err_DIAG', 'MU_SH0ES',
              'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
              'USED_IN_SH0ES_HF', 'c', 'cERR', 'x1', 'x1ERR', 'mB', 'mBERR',
              'x0', 'x0ERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'RA', 'DEC',
              'HOST_RA', 'HOST_DEC', 'HOST_ANGSEP', 'VPEC', 'VPECERR', 'MWEBV',
              'HOST_LOGMASS', 'HOST_LOGMASS_ERR', 'PKMJD', 'PKMJDERR', 'NDOF',
              'FITCHI2', 'FITPROB', 'm_b_corr_err_RAW', 'm_b_corr_err_VPEC',
              'biasCor_m_b', 'biasCorErr_m_b', 'biasCor_m_b_COVSCALE',
              'biasCor_m_b_COVADD'),
    'formats': ['U10'] + ['f8']*46  # 'U10' for the first 'string' column, 'f8' for the numeric columns
    }

    file_path = "./data/Pantheon+SH0ES.dat"
    data = np.genfromtxt(file_path, skip_header=1, dtype=dtype)

    file_path = "./data/Pantheon+SH0ES_STAT+SYS.cov" 
    cov_vec = np.loadtxt(file_path, delimiter=' ') 
    matrix_size = int(np.sqrt(len(cov_vec[1:])))
    cov_matrix = np.reshape(cov_vec[1:], (matrix_size, matrix_size))
    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####
    zvec = data['zHD'].T
    mvec = data['m_b_corr'].T
    muvec = data['MU_SH0ES'].T

    indxs = np.where(zvec>=0.01)
    cov_matrix_fil = cov_matrix[indxs[0],:]
    cov_matrix_fil = cov_matrix_fil[:,indxs[0]]  # Eliminar columnas

    dictfinal = {'mB': mvec[indxs],
                'MU' : muvec[indxs],
                'zHD': zvec[indxs],
                'cov': cov_matrix_fil,
                }

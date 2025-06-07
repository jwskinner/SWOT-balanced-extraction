import numpy as np
import os, sys
sys.path.append(os.path.expanduser('~/scattering_transform/'))
import scattering
import ST

def scattering_func(input_data, J=8, L=4): # Compute the scattering coeficients from a 3D input field 
    M = input_data.shape[1]
    N = input_data.shape[2]
    st_calc = scattering.Scattering2d(M, N, J, L)
    s_mean = st_calc.scattering_coef(input_data)
    S1 = s_mean['S1']
    S2 = s_mean['S2']
    s22 = s_mean['s22']
    s21 = s_mean['s21']
    mean = s_mean['mean']
    var = s_mean['var']
    s_mean['input_field'] = input_data
    s_mean['J'] = J
    s_mean['L'] = L 
    return s_mean
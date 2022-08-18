# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:21:29 2022

@author: DELL
"""

import pandas as pd

compounds = pd.read_csv('data/compound_list.csv', encoding = 'gbk')


# AutoMS
from AutoMS.automs import AutoMS

file = 'data/600mix_pos.mzML'
peaks = AutoMS(file, min_intensity = 500)

# XCMS
xcms = pd.read_csv('data/XCMS_output.csv')

# MS-DIAL
ms_dial = pd.read_csv('data/MS_DIAL_output.csv')

# Peakonly
peakonly = pd.read_csv('data/peakonly_output.csv')
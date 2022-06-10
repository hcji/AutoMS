"""
Module to parse mzML files based on pymzml
"""
# pylint: disable=no-name-in-module
import os
import pymzml
import numpy as np


def readms(file_path):
    """
    Read mzML files.
    Arguments:
        file_path: string
            path to the dataset locally
    Returns:
        Tuple of Numpy arrays: (m/z, intensity, retention time, mean interval of retention time).
    
    Examples:
        >>> from hpic.fileio import readms
        >>> ms,intensity,rt,rt_mean_interval = readms("MM14_20um.mzxml")
    """
    ms_format = os.path.splitext(file_path)[1]
    ms_format = ms_format.lower()
    if ms_format == '.mzml':
        run = pymzml.run.Reader(file_path)
    else:
        raise Exception('ERROR: %s is wrong format' % file_path)
    for n, spec in enumerate(run):
        pass

    m_s = []
    intensity = []
    r_t = []
    for spectrum in run:
        if spectrum.ms_level == 1:
            if spectrum.scan_time[1] == 'minute':
                r_t.append(spectrum.scan_time[0] * 60)
            elif spectrum.scan_time[1] == 'second':
                r_t.append(spectrum.scan_time[0])
            else:
                raise Exception('ERROR: scan time unit is wrong format')
            p_ms = []
            p_intensity = []
            for peak in spectrum.centroidedPeaks:
                if peak[1] != 0:
                    p_ms.append(peak[0])
                    p_intensity.append(peak[1])
            ms_index = np.argsort(np.negative(p_intensity))
            m_s.append(np.array(p_ms)[ms_index])
            intensity.append(np.array(p_intensity)[ms_index])
    rt1 = np.array(r_t)
    if rt1.shape[0] > 1:
        rt_mean_interval = np.mean(np.diff(rt1))
    else:
        rt_mean_interval = 0.0
    return m_s, intensity, r_t, rt_mean_interval
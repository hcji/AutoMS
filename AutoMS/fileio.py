"""
Module to parse mzML files based on pymzml

                            MIT License

Copyright (c) 2019, Zhimin Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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
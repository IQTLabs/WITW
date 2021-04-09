#!/usr/bin/env python

"""
First run convert_strips.py to convert 16bit pansharpened multispectral to 8bit pansharpened RGB.  Then run this script to reproject.
"""

import os
import subprocess

names = [
    '01_rio',
    '02_vegas',
    '03_paris',
    '04_shanghai',
    '05_khartoum',
    '06_atlanta',
    '07_moscow',
    '08_mumbai',
    '09_san',
    '10_dar',
    '11_rotterdam',
]

epsgs = [
    32723,
    32611,
    32631,
    32651,
    32636,
    32616,
    32637,
    32643,
    32620,
    32737,
    32631,
]


for aoi in [6]:
    name = names[aoi-1]
    epsg = epsgs[aoi-1]
    print('AOI:', aoi)
    cmd_string = (
        'gdalwarp'
        + ' -t_srs "EPSG:' + str(epsg) + '"'
        + ' -tr .3 .3'
        + ' -r lanczos'
        + ' -srcnodata None -dstnodata None'
        + ' /local_data/geoloc/sat/psrgb/' + name + '.tif'
        + ' /local_data/geoloc/sat/utm/' + name + '.tif'
    )
    print(cmd_string)
    os.system(cmd_string)
    #subprocess.check_output(cmd_string, shell=True)

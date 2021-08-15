#!/usr/bin/env python

"""
Before or after: Merge preprocessed Rio RGB tiles
gdal_merge.py -o /local_data/geoloc/sat/psrgb/01_rio.tif -of GTiff -v /local_data/geoloc/sat/psms/01_rio/rgb_already_processed/0130222*.tif

After: Generate thumbnails
cd /local_data/geoloc/sat; for file in $(ls psrgb); do echo $file; gdal_translate -outsize 1% 1% psrgb/$file thumbnail/$file; done
"""

import os
import create_8bit_images

for aoi in range(1, 12):
    print('AOI:', aoi)
    if aoi == 1:
        pass # Directly stitch pre-processed RGB tiles with above command
    elif aoi == 2:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/02_vegas/AOI_2_Vegas_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/02_vegas.tif',
            band_order=[5,3,2]
        )
    elif aoi == 3:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/03_paris/AOI_3_Paris_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/03_paris.tif',
            band_order=[5,3,2]
        )
    elif aoi == 4:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/04_shanghai/AOI_4_Shanghai_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/04_shanghai.tif',
            band_order=[5,3,2]
        )
    elif aoi == 5:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/05_khartoum/AOI_5_Khartoum_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/05_khartoum.tif',
            band_order=[5,3,2]
        )
    elif aoi == 6:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/06_atlanta/AOI_6_Atlanta_Atlanta_nadir7_catid_1030010003D22F00_058222361010_01_assembly_cog_PS-RGBNIR.tif',
            '/local_data/geoloc/sat/psrgb/06_atlanta.tif',
            band_order=[3,2,1]
        )
    elif aoi == 7:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/07_moscow/AOI_7_Moscow_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/07_moscow.tif',
            band_order=[5,3,2]
        )
    elif aoi == 8:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/08_mumbai/AOI_8_Mumbai_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/08_mumbai.tif',
            band_order=[5,3,2]
        )
    elif aoi == 9:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/09_san/AOI_9_San_Juan_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/09_san.tif',
            band_order=[5,3,2]
        )
    elif aoi == 10:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/10_dar/AOI_10_Dar_Es_Salaam_PS-MS_COG.tif',
            '/local_data/geoloc/sat/psrgb/10_dar.tif',
            band_order=[5,3,2]
        )
    elif aoi == 11:
        create_8bit_images.convert_to_8Bit(
            '/local_data/geoloc/sat/psms/11_rotterdam/19AUG31104407-Merge_MS-PS.tif',
            '/local_data/geoloc/sat/psrgb/11_rotterdam.tif',
            band_order=[3,2,1]
        )

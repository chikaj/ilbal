#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:12:45 2018

@author: nate
"""

import geopandas
import rasterio
from rasterio import features
from rasterio.plot import show, show_hist
from rasterstats import zonal_stats
from ilbal.obia import data_preparation as dp
from skimage.measure import regionprops
import numpy as np
from pprint import pprint

raster = rasterio.open("../../../../../projects/learn_python/3435.tif")
vector = geopandas.read_file("../../../../../projects/learn_python/3435_segs.shp")

image = raster.read([1, 2, 3])
#show_hist(image, bins=50, lw=0.0, stacked=False, alpha=0.3, histtype='stepfilled')
#show(image)
mask = raster.read_masks(1)
mask[mask == 255] = 1

pprint(vector.crs)
pprint(raster.crs.to_dict())

segmented_image = dp.slic_segmentation(image, mask)
pprint(segmented_image.shape)
#shps = features.shapes(segmented_image.astype(np.int32), connectivity=4, transform=raster.transform)
#shapes = list(shps)
r2vec = dp.ras2vec(segmented_image, raster.transform)


rs = zonal_stats(r2vec, raster.read(1), affine=raster.transform)
pprint(rs)

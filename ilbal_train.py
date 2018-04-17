# -*- coding: utf-8 -*-
"""
Loop through the images to train them for classification.

@author: Nate Currit
"""
from ilbal.obia import data_preparation as dp
from ilbal.obia import utility, classify as cl
import rasterio
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import time


def main():
    ahora = time.time()
    input_training_polygons = "/home/nate/Documents/Research/Guatemala/geobia_training/training_samples2.shp"
    training_segs_output = "/home/nate/Documents/Research/Guatemala/guat_obia/segs/training_segments_felz.shp"
    model_output = "/home/nate/Documents/Research/Guatemala/guat_obia/felz_model"
    
    image_list = utility.get_training_data_as_list()
    base = '/home/nate/Documents/Research/Guatemala/training_data/training_'
#    image_list = ['/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3434.tif',
#                  '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3435.tif',
#                  '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3436.tif']
#    
#    image_list = ['/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3451.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3661.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_4331.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3746.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3411.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3499.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3456.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3460.tif',
#                  '/home/nate/Documents/Research/Guatemala/training_data/training_new_IMG_3414.tif']
    
    first = True
    for i in image_list:
        f = base + i + ".tif"
        with rasterio.open(f) as src:
            image = src.read(masked=True)
            mask = src.read_masks(1)

            vec_temp = dp.prep_for_felz(image, mask, src.transform)
            vec_temp.crs = src.crs.to_dict()
            if first:
                vec = vec_temp
            else:
                vec = pd.concat([vec, vec_temp])
        first = False
            
    full_training = gpd.read_file(input_training_polygons)
    training = gpd.sjoin(vec, full_training, how="inner", op="intersects")
    
    fields = ['count', 'perimeter', 'eccentrici', 'equal_diam', 'major_axis',
              'minor_axis', 'orientatio', 'red_mean', 'green_mean',
              'blue_mean', 'sobel_mean', 'sobel_std']

    segs = training[fields]
    actual = training["class_id"]
    cl.train_classifier(segs, actual, model_output)
    
    gdf.to_file(vec, training_segs_output.replace(".shp", "_all.shp"))
    gdf.to_file(training, training_segs_output)
    
    utility.time_elapsed(ahora)
    

if __name__ == "__main__":
    main()
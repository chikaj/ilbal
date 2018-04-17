# -*- coding: utf-8 -*-
"""
Loop through the images to classify them.

@author: Nate Currit
"""
from ilbal.obia import data_preparation as dp, classify as cl
from ilbal.obia import verification as v
import rasterio
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import pickle
from glob import glob


def main():
    model_path = "/home/nate/Documents/Research/Guatemala/guat_obia/felz_model"
    model = pickle.load(open(model_path, "rb"))
    print(model)
    
    image_path = "/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/"
    
    image_list = glob(image_path + "*.tif")
    
    image_list = ['/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3434.tif',
                  '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3435.tif',
                  '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3436.tif']
    
#    image_list = ['/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3435.tif']
    
    verif = '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/verification.shp'

        
    for i in image_list:
        with rasterio.open(i) as src:
            image = src.read(masked=True)
            mask = src.read_masks(1)

            vec = dp.prep_for_felz(image, mask, src.transform)

        gdf.to_file(vec, i.replace(".tif", "_segs_f.shp"))
        
        fields = ['count', 'perimeter', 'eccentrici', 'equal_diam',
                  'major_axis', 'minor_axis', 'orientatio', 'red_mean',
                  'green_mean', 'blue_mean', 'sobel_mean', 'sobel_std']
#        tmp = vec.loc[:, fields]
        tmp = vec[fields]
        to_predict = tmp.values
        predictions = cl.predict(model, to_predict)
        
        vec['pred'] = predictions
        vec.crs = src.crs.to_dict()

        verification = gpd.read_file(verif)
        full = gpd.sjoin(vec, verification, how="inner", op="within")
        
        cls = full['class_id'].values
        prd = full['pred'].values
        o, p, u, k = v.accuracy(v.cross_tabulation(cls, prd, 11))
        
        print("--------------------------------------------------")
        print()
        print("For image " + i + ", the accuracy metrics are:")
        print("\tOverall Accuracy: " + str(o) + "%")
        print("\tProducers Accuracy: " + str(p) + "%")
        print("\tUsers Accuracy: " + str(u) + "%")
        print("\tKappa Coefficient: " + str(k) + "%")
        print()
        
        gdf.to_file(full, i.replace(".tif", "_verify_segs.shp"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:49:31 2018

@author: nate
"""

from . import utility
import rasterio
from rasterio import features
from geopandas import GeoDataFrame
from affine import Affine
from glob import glob
import numpy as np
from collections import OrderedDict
from rasterstats import zonal_stats
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed
from skimage.measure import regionprops, label
from skimage.filters import sobel


def is_working():
    print("data_preparation.py is working!")


def shift_images(input_directory_path, output_directory_path):
    """
    Shift the images' coordinates.

    This function shifts the image coordinates and removes the 4th Alpha band
    from the photograph. It is used to center photographs from distant
    locations to a central location for manual digitizing for purposes of
    training an image classifier.

    Parameters
    ----------
    input_directory_path: string
        string representing filesystem directory where the photographs are
        located.
    
    output_directory_path: string
        string representing the filesystem directory where the output
        photographs will be written to disk.

    Returns
    -------
    NADA

    """
    x_coord = 476703
    y_coord = 1952787
    x_increment = 350
    y_increment = 350
    x_count = 0
    y_count = 0
    training_data = utility.get_training_data()
    for key in training_data:
        y_shift = y_count * y_increment
        for f in training_data[key]:
            with rasterio.open(input_directory_path + f + ".tif") as src:
                x_shift = x_count * x_increment
                af = src.transform
                meta = src.meta
                meta['count'] = 3
                meta['transform'] = Affine.translation(x_coord + x_shift,
                                    y_coord - y_shift) * Affine.scale(af[0],
                                    af[4])
                with rasterio.open(output_directory_path + "training_" + f + ".tif", "w", **meta) as dst:
                    dst.write(src.read((1, 2, 3)))
            x_count += 1
        y_count += 1
        x_count = 0


def merge_training_data(directory_path):
    """
    Merge all training photographs.

    Merge all training photographs into a single image.
    This is useful as a precursor to extracting raster
    statistics from a raster and then training a classifier.

    Parameters
    ----------
    none

    Returns
    -------
    success
        String

    """
    photos = glob.glob(directory_path + "*.tif")
    photo_readers = []
    for photo in photos:
        photo_readers.append(rasterio.open(photo))

    output = rasterio.merge.merge(photo_readers)
    meta = photo_readers[0].meta
    meta['width'] = output[0].shape[2]
    meta['height'] = output[0].shape[1]
    meta['transform'] = output[1]
    meta['nodata'] = 0
    try:
        with rasterio.open(directory_path + "big_training.tif", "w",
                           **meta) as dst:
            dst.write(output[0])

        return "Success!"
    except Exception:
        return "Failure"


def slic_segmentation(image, mask):
    """
    Segment the image.

    Segment the image using the slic algorithm (from sklearn.segmentation).

    Parameters
    ----------
    image: numpy.array
        A rasterio-style image. Obtained and transformed by:
        src.read(masked=True).transpose(1, 2, 0)

    mask: numpy.array
        A rasterio-style mask. Obtained by src.read_masks(1)
        This function doesn't do anything with mask at the moment.
        This function assumes image has, and is read with, a mask.

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    img = image[0:3].transpose(1, 2, 0)
    segs = slic(img, n_segments=5000, compactness=1, slic_zero=False)
    mask[mask == 255] = 1
    output = segs * mask[np.newaxis, :]

    return label(output)


def slic_zero_segmentation(image, mask):
    """
    Segment the image.

    Segment the image using the slic-0 algorithm (from sklearn.segmentation).

    Parameters
    ----------
    image: numpy.array
        A rasterio-style image. Obtained and transformed by:
        src.read(masked=True).transpose(1, 2, 0)

    mask: numpy.array
        A rasterio-style mask. Obtained by src.read_masks(1)
        This function doesn't do anything with mask at the moment.
        This function assumes image has, and is read with, a mask.

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    img = image[0:3].transpose(1, 2, 0)
    segs = slic(img, n_segments=2500, compactness=0.1, sigma=5, slic_zero=True)
    mask[mask == 255] = 1
    output = segs * mask[np.newaxis, :]

    return label(output)

def felz_segmentation(image, mask):
    """
    Segment the image.

    Segment the image using the felzenszwalb algorithm
    (from sklearn.segmentation).

    Parameters
    ----------
    image: numpy.array
        A rasterio-style image. Obtained and transformed by:
        src.read(masked=True).transpose(1, 2, 0)

    mask: numpy.array
        A rasterio-style mask. Obtained by src.read_masks(1)
        This function doesn't do anything with mask at the moment.
        This function assumes image has, and is read with, a mask.

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    img = image[0:3].transpose(1, 2, 0)
    segs = felzenszwalb(img, scale=50.0, sigma=2, min_size=1500)
    mask[mask == 255] = 1
    output = segs * mask[np.newaxis, :]

    return label(output)


def quickshift_segmentation(image, mask):
    """
    Segment the image.

    Segment the image using the quickshift algorithm
    (from sklearn.segmentation).

    Parameters
    ----------
    image: numpy.array
        A rasterio-style image. Obtained and transformed by:
        src.read(masked=True).transpose(1, 2, 0)

    mask: numpy.array
        A rasterio-style mask. Obtained by src.read_masks(1)
        This function doesn't do anything with mask at the moment.
        This function assumes image has, and is read with, a mask.

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    img = image[0:3].transpose(1, 2, 0)
    segs = quickshift(img, ratio=0.3, kernel_size= 2, max_dist=50, sigma=5)
    mask[mask == 255] = 1
    output = segs * mask[np.newaxis, :]

    return label(output)


def ras2vec(classified_image, transform):
    classified_image = classified_image[0]
    props = regionprops(classified_image)

    shps = features.shapes(classified_image.astype(np.int32), connectivity=4,
                           transform=transform)
    shapes = list(shps)
    records = []

    count = 1
    for shp in shapes:
        if shp[1] != 0:
            id = int(shp[1]-1)
            count = np.int(props[id].area)
            perimeter = np.float(props[id].perimeter)
            eccentricity = np.float(props[id].eccentricity)
            equal_diam = np.float(props[id].equivalent_diameter)
            major_axis = np.float(props[id].major_axis_length)
            minor_axis = np.float(props[id].minor_axis_length)
            orientation = np.float(props[id].orientation)

            item = {'geometry': shp[0],
                    'id': count,
                    'properties': OrderedDict([('DN', np.int(shp[1])),
                                               ('count', count),
                                               ('perimeter', perimeter),
                                               ('eccentrici', eccentricity),
                                               ('equal_diam', equal_diam),
                                               ('major_axis', major_axis),
                                               ('minor_axis', minor_axis),
                                               ('orientatio', orientation)]),
                    'type': 'Feature'}
            records.append(item)
        count += 1

    #return records
    #columns = list(records[0]["properties"]) + ["geometry"]
    return GeoDataFrame.from_features(records)


def edge_detect(image):
    """
    Performs a Sobel edge detection.

    Performs a Sobel edge detection on a 2D image.
    
    Parameters
    ----------
    image: numpy.array
        A rasterio-style image. The image is the first band (or any single
        band, really) obtained and transformed by:
        image = src.read(masked=True).transpose(1, 2, 0). Then image[0].
        
    Returns
    -------
    numpy.array
        A single band, rasterio-style image.
    """
    edges = sobel(image)
    return edges[np.newaxis, :]


def add_zonal_fields(vector, raster, affine, prefix, band=0, stats=['mean',
                                                                    'min',
                                                                    'max',
                                                                    'std']):
    """
    Adds zonal statistics to an existing vector file.

    Add more documentation.

    """
    raster_stats = zonal_stats(vector, raster[band], stats=stats,
                               affine=affine)
#    for item in raster_stats:
#        items = tuple(item.items())
#        for k, v in items:
#            item[prefix + "_" + k] = v
#            item.pop(k)
#
#    for v, rs in zip(vector, raster_stats):
#        v['properties'] = OrderedDict(v['properties'], **rs)
    
    vals = [[] * i for i in range(0, len(raster_stats[0]))]
    header = [[] * i for i in range(0, len(raster_stats[0]))]
    for item in raster_stats:
        items = tuple(item.items())
        count = 0
        for k, v in items:
            vals[count].append(v)
            header[count] = prefix + "_" + k
            count += 1

    for i in range(0, len(raster_stats[0])):
        vector[header[i]] = vals[i]


def add_field_values(vector, crs, fieldname, values):
    """
    Add a field to a vector polygon file and add the values from a land
    cover classification training polygon dataset.
    """
    vector[fieldname] = values
    vector['properties'] = OrderedDict(vector['properties'], fieldname)
    return


def prep_for_slic(image, mask, transform, crs_dict):
    rout = slic_segmentation(image, mask)
    vout = ras2vec(rout, transform)
    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
                         prefix=p, stats=['mean'])
    edges = edge_detect(image[0])
    add_zonal_fields(vector=vout, raster=edges, band=0,
                     affine=transform, prefix='sobel',
                     stats=['mean', 'std', 'sum'])
    
    return vout


def prep_for_slic_zero(image, mask, transform, crs_dict):
    rout = slic_zero_segmentation(image, mask)
    vout = ras2vec(rout, transform)
    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
                         prefix=p, stats=['mean'])
    edges = edge_detect(image[0])
    add_zonal_fields(vector=vout, raster=edges, band=0,
                     affine=transform, prefix='sobel',
                     stats=['mean', 'std', 'sum'])
    
    return vout


def prep_for_felz(image, mask, transform):
    rout = felz_segmentation(image, mask)
    vout = ras2vec(rout, transform)
    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
                         prefix=p, stats=['mean'])
    edges = edge_detect(image[0])
    add_zonal_fields(vector=vout, raster=edges, band=0,
                     affine=transform, prefix='sobel',
                     stats=['mean', 'std', 'sum'])
    
    return vout


def prep_for_quickshift(image, mask, transform, crs_dict):
    rout = quickshift_segmentation(image, mask)
    vout = ras2vec(rout, transform)
    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
                         prefix=p, stats=['mean'])
    edges = edge_detect(image[0])
    add_zonal_fields(vector=vout, raster=edges, band=0,
                     affine=transform, prefix='sobel',
                     stats=['mean', 'std', 'sum'])
    
    return vout

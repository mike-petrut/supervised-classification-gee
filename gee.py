import os
import ee
import pandas as pd
import numpy as np
import geemap
import geopandas as gpd

Map = geemap.Map(center = [-33.9628184, 117.5364922], zoom = 4)

forest_shp = gpd.read_file('forest-shapefile/forest.shp').to_crs('EPSG:4326')

forest_shp.head(3)

lga = gpd.read_file('lga/lga.shp')

nannup = lga[lga['LGA_NAME20'] == 'Nannup (S)'].explode().to_crs('EPSG:4326')

features = []

for index, gpd_row in nannup.iterrows():
    
    geom = gpd_row.geometry 
    x, y = geom.exterior.coords.xy
    coords = np.dstack((x,y)).tolist()
    ee_geom = ee.Geometry.Polygon(coords)          
    feature = ee.Feature(ee_geom)

    features.append(feature)
            
nannup_ee = ee.FeatureCollection(features)

forest_shp_clip = gpd.clip(forest_shp, nannup)
forest_shp_clip = (forest_shp_clip[forest_shp_clip['map_symbol']
                    .isin(['Hardwood', 'Sandalwood', 'Softwood', 'Other Species'])])

# Input imagery is a cloud-free Landsat 8 composite.
l8 = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        .filterBounds(nannup_ee)
        .filterDate('2021-06-01', '2021-08-31'))

image = ee.Algorithms.Landsat.simpleComposite(**{
  'collection': (l8)
    ,
  'asFloat': True
})

# Use these bands for prediction.
bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

def gpd_to_ee(geoframe, poly_class):

    geoframe = geoframe.explode()

    poly_cat = geoframe[poly_class].unique() 
    poly_num = [1,2,3,4]

    features = []

    for cat, num in zip(poly_cat, poly_num):

        polygons =  geoframe[geoframe[poly_class] == cat]

        for index, gpd_row in polygons.iterrows():

            geom = gpd_row.geometry 
            x, y = geom.exterior.coords.xy
            coords = np.dstack((x,y)).tolist()
            ee_geom = ee.Geometry.Polygon(coords)          
            feature = ee.Feature(ee_geom, {'class': num})

            features.append(feature)
            
    ee_poly = ee.FeatureCollection(features)    

    return ee_poly

forest_ee = gpd_to_ee(forest_shp_clip, 'map_symbol')

# Get the values for all pixels in each polygon in the training.
training = image.sampleRegions(**{
  # Get the sample from the polygons FeatureCollection.
  'collection': forest_ee,
  # Keep this list of properties from the polygons.
  'properties': ['class'],
  # Set the scale to get Landsat pixels in the polygons.
  'scale': 30
})

# Create an SVM classifier with custom parameters.

classifier = ee.Classifier.libsvm(**{
  'kernelType': 'RBF',
  'gamma': 0.5,
  'cost': 10
})

# Train the classifier.
trained = classifier.train(training, 'class', bands)

# Classify the image.
classified = image.classify(trained)

# Display the classification result and the input image.

Map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'max': 0.5, 'gamma': 2})
Map.addLayer(forest_ee, {'min': 1, 'max': 4, 'palette': ['green', 'blue', 'yellow', 'red']}, 'training polygons')
#Map.addLayer(classified, {'min': 1, 'max': 4, 'palette': ['green', 'blue', 'yellow', 'red']}, 'classified')


# Display the map.
path = os.getcwd()
html_file = os.path.join(path, 'forest_map.html')
Map.to_html(outfile = html_file, title = 'Forest Map', width = '100%', height = '880px')



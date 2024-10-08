"""
Script for calculating distance to shore for locations with temperature logger data
Uses outputs from scripts in logger_processing folder
Author: Jaelyn Bos
2024
"""

'Import libraries'
import numpy as np
import pandas as pd
import geopandas as gpd
import xrspatial as xrs
from geocube.api.core import make_geocube
from shapely.geometry import Polygon

'Path to folder where logger data is saved'
path = '/home/jtb188/'

'Path to folder whre shoreline data is saved'
shore_path = '/projects/f_mlp195/bos_microclimate/'

'Path to folder where logger data with calculated distances will be saved'
out_path = '/projects/f_mlp195/bos_microclimate/'

'Import shoreline data. Data is open source line map from OpenStreetMap project'
shoreline = gpd.read_file(shore_path + 'OpenStreetMap_shoreline/lines.shp')

'Remove data outside the tropics'
tropic_mask = Polygon([(-180,-35),(-180,35),(180,35),(180,-35)])
shoreline = gpd.clip(shoreline,tropic_mask)
shoreline['SHORE'] = 1

'Convert shoreline data to raster (geocube), define resolution and CRS based on metadata'
shorecube =  make_geocube(vector_data=shoreline, output_crs="EPSG:4326",resolution=(0.001,0.001), fill=0)['SHORE']

'Use proximity function from the xarray-spatial packages to calculate the great circle distance between each non-shoreline pixel and the nearest shoreline'
shoredist = xrs.proximity(shorecube,distance_metric='GREAT_CIRCLE')

'Import logger data and add region column before concatenating'
aims = pd.read_csv(path + 'aims.csv')
aims['REGION'] = 'AUSTRALIA'
noaa_caribbean = pd.read_csv(path + 'noaa_caribbean.csv')
noaa_caribbean['REGION'] = 'CARIBBEAN'
noaa_hawaii = pd.read_csv(path + 'noaa_hawaii.csv')
noaa_hawaii['REGION'] = 'HAWAII'
noaa_pacific = pd.read_csv(path + 'noaa_pacific.csv')
noaa_pacific['REGION'] ='NOAA_PACIFIC'
palau = pd.read_csv(path + 'palau.csv')
palau['REGION'] = 'PALAU'

reefTEMPS = pd.read_csv(path + 'reeftemps.csv')
reefTEMPS['REGION'] = 'NEW_CALEDONIA'
panama = pd.read_csv(path + 'panama.csv')
panama['REGION'] = 'PANAMA'
kiritimati = pd.read_csv(path + 'kiritimati.csv')
kiritimati['REGION'] = 'KIRITIMATI'
dongsha = pd.read_csv(path + 'dongsha.csv')
dongsha['REGION'] = 'DONGSHA'
pipa = pd.read_csv(path + 'pipa.csv')
pipa['REGION'] = 'PHOENIX ISLANDS'

training_temps = pd.concat([aims,noaa_caribbean,noaa_hawaii,noaa_pacific,palau],axis=0,ignore_index=True)
training_temps = training_temps.reset_index(drop=True)

'Calculte distance to shore for each logger by finding the nearest pixel centroid of the distance surface to each logger lat/lon pair'
dists = []
for i in range(0,len(training_temps)):
    row = training_temps.iloc[i,:]
    x = (shoredist.sel(x = row['LONGITUDE'], y = row['LATITUDE'], method="nearest"))
    y = x.values.item()
    dists.append(y)

'Add SHOREDIST column, with distance to shore in meters, to the logger dataset and export as csv'
training_temps['SHOREDIST'] = pd.Series(dists) 
training_temps.to_csv(out_path + 'training_temps.csv')
del dists

'Repeat process for the rest of the loggers'
testing_temps = pd.concat([reefTEMPS,panama,dongsha,kiritimati,pipa],axis=0,ignore_index=True)
testing_temps = testing_temps.reset_index(drop=True)

dists = []
for i in range(0,len(testing_temps)):
    row = testing_temps.iloc[i,:]
    x = (shoredist.sel(x = row['LONGITUDE'], y = row['LATITUDE'], method="nearest"))
    y = x.values.item()
    dists.append(y)
    
testing_temps['SHOREDIST'] = pd.Series(dists) 
testing_temps.to_csv(out_path + 'testing_temps.csv')
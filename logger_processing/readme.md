Scripts in this directory should be run before scripts in the remote_data_calcs directory. 

Scripts in this directory import raw temperature logger data (in a mix of .csv, .mat, and .netcdf formats) and export .csv files with summarized data for each point. 
Each script named with a region or data source (AIMS, Dongsha, Kiritimati, NOAA/USGS,Palau,phoenix\_islands,stri\_panama,and the reefTEMPS network on New caledonia) imports raw data from that region. Each script does the following:
    The data is formatted, including a date time variable in datetime(64) format, and data prior to 2003 or after January of 2020 are deleted as necessary. 
    The logger region is merged with geomorphic data from the Allen Coral Atlas to add geomorphic class, and the points outside the range of the geomorphic data are discarded. 
    Loggers without data from at least twelve months are discarded. 
    Max monthly means and daily temperature ranges are calculated, and aggregating data into one line per point. 
    Formatted data are exported to .csv files.
    
shoredist_merge.py takes formatted .csv files as input. This script calculates distance to shore in meters from OpenStreetMap data (downloaded 10 March 2023) for each point. 
It then aggregates all of the data into two files: training\_temps.csv and testing\_temps.csv. These files can be used to get sea surface temperature data from the NASA server, then used for model fitting.
The 'testing' and 'training' format is an artifact of an earlier version of this study. All of the points are combined into a single dataset at the analysis stage. 

Geomorphic data for each region can be downloaded from AllenCoralAtlas.org/atlas. As the Allen Coral Atlas updates routinely, copies of the exact geomorphic class datasets used in this analysis can be found on Figshare.

OpenStreetMap coastlines data can be downloaded from https://osmdata.openstreetmap.de/data/coastlines.html. OpenStreetMap data is available under an OpenDatabaseLicense, ODbL 1.0. For more information see https://osmfoundation.org/wiki/Licence

Scripts in this directory require the following libraries:\
numpy 1.23.5\
pandas 1.4.2\
geopandas 1.10.2\
scipy 1.11.4\
shapely 1.8.2\
xarray 2023.10.1

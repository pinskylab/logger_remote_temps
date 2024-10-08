Jaelyn Bos
23 March 2023

Scripts in this directory import raw temperature logger data (in a mix of .csv, .mat, and .netcdf formats) and export .csv files with summarized data for each point. 
Each script named with a region or data source (AIMS, dongsha, kiritimati, NOAA,palau,phoenix\_islands,stri\_panama) imports raw data from that region. Each script does the following:
    The data is formatted, including a date time variable in datetime(64) format, and data prior to 2003 are deleted as necessary. 
    The logger region is merged with geomorphic data from the Allen Coral Atlas to add geomorphic class, and the points outside the range of the geomorphic data are discarded. 
    Loggers without data from at least twelve months are discarded. 
    Max monthly means and daily temperature ranges are calculated, and aggregating data into one line per point. 
    Formatted data are exported to .csv files.
    
shoredist_merge.py takes formatted .csv files as input. This script calculates distance to shore in meters from OpenStreetMap data (downloaded 10 March 2023) for each point. 
It then aggregates all of the data into two files: training\_temps.csv and testing\_temps.csv. These files can be used to get sea surface temperature data from the NASA server, then used for model fitting.

sst_sel.py must be run on the NASA Amazon Web Services server for Multi-Scale Ultra High Resolution Sea Surface Temperature at https://registry.opendata.aws/mur/
This repo contains scripts used in data analysis for the manuscript 
*Fine resolution satellite sea surface temperatures capture the conditions experienced by corals at monthly but not daily time scales* (Bos & Pinsky). 

Repo contents:
Logger_processing folder contains scripts to standardize data and calculate temperature metrics for various *in situ* temperature logger datasets. The scripts in this folder should be run first. 

Remote_data_calcs folder contains scripts to calculate temperature metrics from NASA/JPL Multi-scale Ultra-high Resolution (MUR) Sea Surface Temperature dataset. The scripts in this folder should be run second. 

Analysis_and_figs folder contains scripts to run all linear models, make all figures in the manuscript and supplementary material, and calculate other miscellaneous stats 
(Pearson's r, Kruskil-Wallis F, etc.) mentioned in the manuscript. 
This folder also contains a copy of the fully processed logger data with in situ and remotely sensed temperature metrics in .csv form. 
Therefore, the scripts in this folder can be run third, but can also be run independently, since the fully processed data is included in the folder. 

All code by Jaelyn Bos, 2024

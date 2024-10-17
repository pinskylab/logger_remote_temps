This folder contains scripts for all figures, linear models, and statistical analysis, as well as a copy of the fully processed logger data with in situ and remotely sensed temperature metrics in .csv form. 
The logger data is divided into "testing" and "training" data from an earlier version of this analysis, but the files get combined within the scripts. 

figures.py can be used to recreate all of the figures in the manuscript and supplementary material.
linear models.py can be used to run linear and random forest models relating various types of remotely sensed data to in situ temperature metrics. This can be used to recreate the models detailed in the supplementary material, or experiment with different combinations of variables.
KruskallWallis and misc.py can be used to recreate the Kruskal-Wallis tests, linear correlations, and summary statistics described in the manuscript. 

Files in this folder require the following libraries:\
cartopy 0.22.0\
geopy 2.4.1\
matplotlib 3.7.1\
numpy 1.23.5\
pandas 1.4.2\
scikit-learn 1.0.2\
scipy 1.11.4\
seaborn 0.11.2

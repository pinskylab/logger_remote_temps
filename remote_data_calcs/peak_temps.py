'''
Author: Jaelyn Bos
2024
'''

'''
This script takes as input logger data with calculated MMMs from remote data and outputs logger data with remote MMMs and max temperatures
'''
'Import libraries'
import numpy as np
import pandas as pd
import xarray as xr
from numba import jit
import dask

'Filepath for input logger data'
path = '/projects/f_mlp195/bos_microclimate/'

'Filepath for output data'
out_path = '/projects/f_mlp195/bos_microclimate/'

@jit
def find_max(row):
    sst0 = sst.where(sst['time'] > pd.to_datetime(row['START_TIME']).to_datetime64(),drop=True)
    sst0 = sst0.where(sst['time'] < pd.to_datetime(row['END_TIME']).to_datetime64(),drop=True)
    sst0 = sst0.chunk(dict(time=-1))
    sst0 = sst0.groupby('time.month').max()
    m99 = sst0.max(dim='month')
    a = m99.sel(lat = row['LATITUDE'], lon = row['LONGITUDE'] , method= "nearest")
    a = a.values.item()
    if (a > 0):
        return(a)
    else:
        sst_i = m99.where(sst0['lat'] < row['LATITUDE']+ 0.01, drop=True)
        sst_i = sst_i.where(sst_i['lat'] > row['LATITUDE']- 0.01, drop=True)
        sst_i = sst_i.where(sst_i['lon'] < row['LONGITUDE']+ 0.01, drop=True)
        sst_i = sst_i.where(sst_i['lon'] > row['LONGITUDE']-0.01, drop=True)
        b = sst_i.mean()
        b = b.values.item()
        if (b > 0):
            return(b)
        else:        
            sst_i = m99.where(sst0['lat'] < row['LATITUDE']+ 0.02, drop=True)
            sst_i = sst_i.where(sst_i['lat'] > row['LATITUDE']- 0.02, drop=True)
            sst_i = sst_i.where(sst_i['lon'] < row['LONGITUDE']+ 0.02, drop=True)
            sst_i = sst_i.where(sst_i['lon'] > row['LONGITUDE']-0.02, drop=True)
            c = sst_i.mean()
            return(c.values.item())

'''
Opens SST data hosted on AWS, paid by NASA. Data is in Zarr format for cloud processing
'''
ds_sst = xr.open_zarr('https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1',consolidated=True)
sst = ds_sst['analysed_sst']

'''
Drop data from before 2003, and outside the tropics 
'''
sst = sst.where(sst['time']>pd.to_datetime('2003-01-01'),drop=True)
sst = sst.where(sst['lat']>-32,drop=True)
sst = sst.where(sst['lat']<32,drop=True)

'''
Reading in testing and training datasets
'''
testing = pd.read_csv(path+ 'testing_temps_sst.csv')
testing = testing.drop(columns = 'Unnamed: 0')
testing = testing.reset_index(drop=True)

training = pd.read_csv(path+'training_temps_sst.csv')
training = training.drop(columns = 'Unnamed: 0')
training = training.reset_index(drop=True)

'''
Set up calculation with dask_delayed
'''
ssts_calc = []

def return_ssts(m,n):
    for i in range(m,n):
        lazy = dask.delayed(find_max)(training.iloc[i,:])
        ssts_delayed.append(lazy)
        
'''
Do delayed calculations in batches of 8 - balance between speed and overloading RAM
Training points
'''
for i in range(0,training.shape[0]//8):
    ssts_delayed = []
    m = i*8
    n = i*8 + 8
    return_ssts(m,n)    
    y = list(dask.compute(ssts_delayed,allow_rechunk=True))[0]
    ssts_calc = ssts_calc + y

'''
Batch calculate remaining points not divisible by 8
Training points
'''
ssts_delayed = []
m = (training.shape[0]//8)*8
n = training.shape[0]
return_ssts(m,n)    
y = list(dask.compute(ssts_delayed,allow_rechunk=True))[0]
ssts_calc = ssts_calc + y

'''
Write outfiles for training points
'''
training_ssts = pd.Series(ssts_calc)

training_temps_sst =training
training_temps_sst['SST_MAX'] = training_ssts- 273.15

outfile = open(out_path + 'training_temps_max.csv','wb')
training_temps_sst.to_csv(outfile)
outfile.close()

del training, training_temps_sst, ssts_delayed, ssts_calc, training_ssts
'''
Do delayed calculations in batches of 8 - balance between speed and overloading RAM
Testing points
'''
ssts_calc = []

def return_ssts(m,n):
    for i in range(m,n):
        lazy = dask.delayed(find_max)(testing.iloc[i,:])
        ssts_delayed.append(lazy)

for i in range(0,testing.shape[0]//8):
    ssts_delayed = []
    m = i*8
    n = i*8 + 8
    return_ssts(m,n)    
    y = list(dask.compute(ssts_delayed,allow_rechunk=True))[0]
    ssts_calc = ssts_calc + y


'''
Batch calculate remaining points not divisible by 8
Testing  points
'''
ssts_delayed = []
m = (testing.shape[0]//8)*8 
n = testing.shape[0]
return_ssts(m,n)    
y = list(dask.compute(ssts_delayed,allow_rechunk=True))[0]
ssts_calc = ssts_calc + y

'''
Write outfiles for testing points
'''
testing_ssts = pd.Series(ssts_calc)
testing_temps_sst =testing
testing_temps_sst['SST_MAX'] = testing_ssts - 273.15

outfile = open(out_path + 'testing_temps_max.csv','wb')
testing_temps_sst.to_csv(outfile)
outfile.close()

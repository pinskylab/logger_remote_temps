'''
Jaelyn Bos
September 2023
'''
import numpy as np
import pandas as pd
import xarray as xr
from numba import jit
import dask

'''
Define function to find SSTs. @jit is numba decorator to attempt to compile function to machine code. 
Doesn't work perfectly here, since numba can't handle xarray or pandas specific functions, 
but it does give a slight speedup (checked with timeit)
Outer loop of function finds closest SST value to logger point and checks if that value is NA
If the value is not NA, it gets returned. If the value is NA:
    the function takes the four closes pixels and averages them. 
    If at least one of the pixels is *not* an NA, this will produce a real number
    The function checks if the average value is NA. If it is *not* NA, it gets returned. If it is NA:
        the function takes the nine closest pixels and averages them. That number gets returned without an NA check
'''
@jit
def find_sst(row):
    sst0 = sst.where(sst['time'] > pd.to_datetime(row['START_TIME']).to_datetime64(),drop=True)
    sst0 = sst0.where(sst['time'] < pd.to_datetime(row['END_TIME']).to_datetime64(),drop=True)
    sst0 = sst0.groupby('time.month').mean('time',keep_attrs=True,skipna=False)
    mmm0 = sst0.max(dim='month')
    a = mmm0.sel(lat = row['LATITUDE'], lon = row['LONGITUDE'] , method= "nearest")
    a = a.values.item()
    if (a > 0):
        return(a)
    else:
        sst_i = mmm0.where(sst0['lat'] < row['LATITUDE']+ 0.01, drop=True)
        sst_i = sst_i.where(sst_i['lat'] > row['LATITUDE']- 0.01, drop=True)
        sst_i = sst_i.where(sst_i['lon'] < row['LONGITUDE']+ 0.01, drop=True)
        sst_i = sst_i.where(sst_i['lon'] > row['LONGITUDE']-0.01, drop=True)
        b = sst_i.mean()
        b = b.values.item()
        if (b > 0):
            return(b)
        else:        
            sst_i = mmm0.where(sst0['lat'] < row['LATITUDE']+ 0.02, drop=True)
            sst_i = sst_i.where(sst_i['lat'] > row['LATITUDE']- 0.02, drop=True)
            sst_i = sst_i.where(sst_i['lon'] < row['LONGITUDE']+ 0.02, drop=True)
            sst_i = sst_i.where(sst_i['lon'] > row['LONGITUDE']-0.02, drop=True)
            c = sst_i.mean()
            return(c.values.item())

'Filepath for input logger data'
path = '/projects/f_mlp195/bos_microclimate/'

'Filepath for output data'
out_path = '/projects/f_mlp195/bos_microclimate/'

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
testing = pd.read_csv(path+'testing_temps.csv')
testing = testing.drop(columns = ['Unnamed: 0.1','Unnamed: 0'])
testing = testing.reset_index(drop=True)

training = pd.read_csv(path+'training_temps.csv')
training = training.drop(columns = ['Unnamed: 0.1','Unnamed: 0'])
training = training.reset_index(drop=True)

'''
Set up calculation with dask_delayed
'''
ssts_calc = []

def return_ssts(m,n):
    for i in range(m,n):
        lazy = dask.delayed(find_sst)(training.iloc[i,:])
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
training_temps_sst['SST'] = training_ssts- 273.15

outfile = open(out_path + 'training_temps_sst.csv','wb')
training_temps_sst.to_csv(outfile)
outfile.close()
'''
Do delayed calculations in batches of 8 - balance between speed and overloading RAM
Testing points
'''

ssts_calc = []

def return_ssts(m,n):
    for i in range(m,n):
        lazy = dask.delayed(find_sst)(testing.iloc[i,:])
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
testing_temps_sst['SST'] = testing_ssts - 273.15

outfile = open(out_path+'testing_temps_sst.csv','wb')
testing_temps_sst.to_csv(outfile)
outfile.close()
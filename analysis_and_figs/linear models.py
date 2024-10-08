"""
Author:Jaelyn Bos
"""
'''
Import packages
'''
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import scipy.stats as stats

'''
Load data
'''
'Path to saved logger data'
data_path = '/projects/f_mlp195/bos_microclimate/'

train_dat = pd.read_csv(data_path + '/training_weeklydiffs_sst.csv')
test_dat = pd.read_csv(data_path +'testing_weeklydiff_sst.csv')

'Combine datasets'
all_dat = pd.concat([train_dat,test_dat])
all_dat = all_dat.reset_index(drop=True)

'Take absolute value of latitude'
all_dat['ABSLAT'] = all_dat['LATITUDE'].abs()

'Take natural log of distance to shore'
all_dat['LOGDIST'] = np.log1p(all_dat['SHOREDIST'])

'Create dummy variable of different geomorphic classes'
rclass = pd.get_dummies(all_dat['RCLASS'])

'Create dependent variable to model fitting'
y = pd.DataFrame(all_dat['TEMP_C'])

'''
Define model types
'''
p = PolynomialFeatures(interaction_only=True)

def mod_compare(x,y,mod_type):
    if mod_type == 'lm':
        lm = linear_model.LinearRegression()
        lm.fit(x,y)
        lm_preds = lm.predict(x)
        r2 = sklearn.metrics.r2_score(y,lm_preds)
        cv = cross_val_score(lm,x,y,scoring='r2',cv=5).mean()
        coefficients = lm.coef_
        return("r2=",r2,"crossval_r2=",cv,"lm_coef=",coefficients,"intercept=",lm.intercept_)
    if mod_type == 'rf':
        rf = RandomForestRegressor(n_estimators=100,min_samples_leaf = 3).fit(x,y)
        rf_preds = rf.predict(x)
        r2 = sklearn.metrics.r2_score(y,rf_preds)
        cv = cross_val_score(rf,x,y,scoring='r2',cv=5).mean()
        importances = rf.feature_importances_
        return("train_r2=",r2,"crossval_r2=",cv,"var_importances=",importances)
    else:
        return('invalid model type')

'''
Modeling max monthly means
'''
'Remote MMM only'
x = pd.DataFrame(all_dat['SST'])
mod_compare(x,y,'lm')

'Remote MMM and latitude'
x = all_dat.loc[:,['SST','ABSLAT']]
mod_compare(x,y,'lm')

'Remote MMM and shore distance'
x = all_dat.loc[:,['SST','LOGDIST']]
mod_compare(x,y,'lm')

'Remote MMM and shore distance + interaction lm'
x = pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']]))
x = x.drop(columns=0)
mod_compare(x,y,'lm')

'Remote MMM, shore distance, latitude random forest'
x = p.fit_transform(all_dat.loc[:,['SST','LOGDIST','LATITUDE']])
mod_compare(x,y,'rf')

'Remote MMM, shore distance, interaction, and all geomorphic classes random forest'
x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'rf')

'Remote MMM, shore distance, interaction, and all geomorphic classes linear model'
x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

'Remote MMM, shore distance, interaction, each individual geomorphic class linear models. Mean cross-validated R2 below'

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')
#8816

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Back Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Sheltered Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Inner Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Terrestrial Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Reef Crest']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Deep Lagoon']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Shallow Lagoon']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Plateau']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

'Remote MMM, shore distance, interaction, and best performing geomorphic class (outer reef flat) with each other geomorphic class'

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Back Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')
#8819

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Sheltered Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Inner Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Terrestrial Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Reef Crest']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Deep Lagoon']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Plateau']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

'Remote MMM, shore distance, interaction, outer reef flat, and shallow lagoon with each other geomorphic class'

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Back Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Sheltered Reef Slope']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Inner Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Terrestrial Reef Flat']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Reef Crest']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Deep Lagoon']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

x = pd.concat((pd.DataFrame(p.fit_transform(all_dat.loc[:,['SST','LOGDIST']])),rclass[['Outer Reef Flat', 'Shallow Lagoon', 'Plateau']]),axis=1)
x = x.drop(columns=0)
mod_compare(x,y,'lm')

'''
Modeling peak temps
'''
y = pd.DataFrame(all_dat['Q99'])

'Remote MMM only'
x = pd.DataFrame(all_dat['SST'])
mod_compare(x,y,'lm')

'Remote peak temp only'
x = pd.DataFrame(all_dat['SST_MAX'])
mod_compare(x,y,'lm')

'Remote peak temp and absolute latitude'
x = all_dat.loc[:,['SST_MAX','ABSLAT']]
mod_compare(x,y,'lm')

'Remote peak temp and shore distance'
x = all_dat.loc[:,['SST_MAX','LOGDIST']]
mod_compare(x,y,'lm')

'Remote MMM and all geomorphic classes linear model'
x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass],axis=1)
mod_compare(x,y,'lm')

'Remote MMM and all geomorphic classes random forest'
x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass],axis=1)
mod_compare(x,y,'rf')

'Remote MMM, log distance to shore, and all geomorphic classes random forest'
x = pd.concat([pd.DataFrame(all_dat.loc[:,['SST_MAX','LOGDIST']]),rclass],axis=1)
mod_compare(x,y,'rf')

'Individual geomorphic classes'

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Back Reef Slope']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Sheltered Reef Slope']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Reef Slope']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Deep Lagoon']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Shallow Lagoon']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Inner Reef Flat']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Outer Reef Flat']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Reef Crest']],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass['Plateau']],axis=1)
mod_compare(x,y,'lm')

'Two geomorphic classes'
x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Back Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Sheltered Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Deep Lagoon']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Shallow Lagoon']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Reef Crest']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Plateau']]],axis=1)
mod_compare(x,y,'lm')

'Three geomorphic classes'

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Back Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Sheltered Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Deep Lagoon']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Shallow Lagoon']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Reef Crest']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Plateau']]],axis=1)
mod_compare(x,y,'lm')

'Four'
x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Shallow Lagoon','Back Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Shallow Lagoon','Sheltered Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Shallow Lagoon','Reef Slope']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Shallow Lagoon','Deep Lagoon']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Shallow Lagoon','Reef Crest']]],axis=1)
mod_compare(x,y,'lm')

x = pd.concat([pd.DataFrame(all_dat['SST_MAX']),rclass[['Inner Reef Flat','Outer Reef Flat','Reef Crest','Plateau']]],axis=1)
mod_compare(x,y,'lm')


'''
Modeling daily temperature range
'''
y = pd.DataFrame(all_dat['DAILY_DIFF'])

'From weekly diff only'
x = pd.DataFrame(all_dat['WEEKLY_DIFF'])
mod_compare(x,y,'lm')

'Remote MMM'
x = pd.DataFrame(all_dat['SST'])
mod_compare(x,y,'lm')

'Remote peak temps'
x = pd.DataFrame(all_dat['SST_MAX'])
mod_compare(x,y,'lm')

'Shore distance'
x = pd.DataFrame(all_dat['LOGDIST'])
mod_compare(x,y,'lm')

'All geomorphic classes'
x = rclass
mod_compare(x,y,'lm')

x = rclass
mod_compare(x,y,'rf')

'Individual geomorphic classes'
x = pd.DataFrame(rclass['Back Reef Slope'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Reef Slope'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Sheltered Reef Slope'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Deep Lagoon'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Shallow Lagoon'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Outer Reef Flat'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Inner Reef Flat'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Terrestrial Reef Flat'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Reef Crest'])
mod_compare(x,y,'lm')

x = pd.DataFrame(rclass['Plateau'])
mod_compare(x,y,'lm')
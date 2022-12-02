import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime,timedelta
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold



# import the grid name

ozonedatapath = "/ozoneinputpath/MDA8O3_grid_deseason_anomaly.csv"

ozone_data = pd.read_csv(
        ozonedatapath
        )

regions_all = np.array(ozone_data.columns[1:len(ozone_data.columns)])

allyears = np.array(np.arange(2015,2020,1))


# output path
ozone_outputpath = "/output/02_RFR_grid/GridOut/"

# met data dir
metdir = "/meteorology/02_RFR_grid/metinput/"




for year_for_test in allyears:
    print(year_for_test)

    for tt in np.arange(0,len(regions_all),1):
    
        region = regions_all[tt]
    
        ozone = pd.read_csv(
                ozonedatapath
                )

        ozonesub = ozone[['date',region]]

        ozonesub = ozonesub.rename(columns={region:"O3"})


        #import the met data

        met_dir = metdir

        metfilenames = os.listdir(met_dir)
        metnames = np.array(metfilenames)

        metnames

        # getting the met variables
        metvars = np.array([])

        for i in np.arange(0,len(metnames),1):
    
            metvar = metnames[i].replace("DS_","").replace("Daily_2015to2019.csv","").replace("_2015to2019.csv","")
    
            metvars = np.append(metvars,metvar)
    
        metvars


        # import the met data:
        # making all the metdataset into a single dataFrame
        # this only works when all dataframe have same row number

        for i in np.arange(0,len(metnames),1):
            met_data = pd.read_csv(met_dir+metnames[i])
            met_val = met_data[region]
    
            if i == 0:
                met_date = met_data["date"]
    
                met_data_all = pd.DataFrame( {'date':met_date})
                met_data_all[metvars[i]] = met_val
            else:
                met_data_all[metvars[i]] = met_val
    
    
    
        #merging with ozone
        
        met_ozone_data = pd.merge(met_data_all,ozonesub,on="date")

        

        # selecting only Apr to October
        met_ozone_data['month'] = pd.DatetimeIndex(met_ozone_data['date']).month
    
        met_ozone_data_sub = met_ozone_data.loc[met_ozone_data['month'].isin(np.arange(4,11,1)),:]

        # drop the whole row when ozone is na

        #drop NA
        met_ozone_data_sub_noNAN = met_ozone_data_sub.dropna()



        met_ozone_data_sub_noNAN.index = np.arange(0,len(met_ozone_data_sub_noNAN.iloc[:,0]),1)
        
        yearcol = pd.DatetimeIndex(met_ozone_data_sub_noNAN['date']).year
    
        yearix = np.where(yearcol == year_for_test)[0]

        met_ozone_data_sub_noNAN_nodatemonth = met_ozone_data_sub_noNAN.drop(["date","month"],axis=1)

        

        # split the data
    
    
        test_dataset = met_ozone_data_sub_noNAN_nodatemonth.iloc[yearix,:]
    
        train_dataset = met_ozone_data_sub_noNAN_nodatemonth.drop(test_dataset.index)


        train_stats = train_dataset.describe()
        train_stats.head()

        train_stats.pop("O3") # pop will drop the column


        train_stats = train_stats.transpose()
        train_stats
    
        # extract label from features
        train_labels = train_dataset.pop('O3')
        test_labels = test_dataset.pop('O3')

        # standardize/normalize the data
        # see: https://www.tensorflow.org/tutorials/keras/regression
        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        normed_train_data = norm(train_dataset)
        normed_test_data = norm(test_dataset)


        # only need to normalize the features, no need to normalize for Y in RFR
        # RFR is more flexible

        normed_train_data.head()
        normed_test_data.head()
        
        # -------------------------- gridSearchCV defining the matrix for hyperparameters ---------------
    
        param_grid = {
                'bootstrap': [True],
                'max_depth': [8,9,10,11,12,13,14,15],
                'max_features': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], # it is fraction here
                'min_samples_leaf': [3],
                'max_samples': [0.3,0.4,0.5,0.6,0.7,0.8], # it is fraction here
                'n_estimators': [200]
                }


        #--------------------------- fitting the model ---------------------------------------------
        
    
        rf = RandomForestRegressor()
        
        cv = KFold(n_splits=4,random_state=None, shuffle=False) # 4 years left for training, 2015 to 2019 is 5-year, test is taken out for a year
        
        
        trainvalidationSplit = list(cv.split(normed_train_data)).copy() # you can see the index assigning to training and validation data
        # we can assign it following the tutorial from: https://stackoverflow.com/questions/46718017/k-fold-in-python
        
        
        date_for_train = met_ozone_data_sub_noNAN.drop(test_dataset.index)
        
        date_for_train.index = np.arange(0,len(date_for_train.iloc[:,0]),1)
        
        date_for_train['year'] = pd.DatetimeIndex(date_for_train['date']).year
        
        len(date_for_train) == len(normed_train_data)
        
        
        training_year = date_for_train['year'].unique()
        
        # assigning the index for cv split
        # because NA occurs in the dataset, we need to make sure that each fold contains one year of data
        for cvindex in np.arange(0,len(date_for_train['year'].unique()),1):
           # print(cvindex)
            validationYear = training_year[cvindex]
            validationix = np.array(date_for_train['year'][date_for_train['year'] == validationYear].index)
            trainingix = np.array(date_for_train['year'].drop(date_for_train['year'][date_for_train['year'] == validationYear].index).index)
            
            tu_obj = (trainingix,validationix)
           
            trainvalidationSplit[cvindex] = tu_obj
            
   
    
    
        
        
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = trainvalidationSplit, n_jobs = -1, verbose = 2) 
        # default using sklearn.metrics.r2_score
    

        grid_search.fit(normed_train_data, train_labels)
    
        grid_search.best_params_
        
        
    
        tree_number = grid_search.best_params_['n_estimators']
        maxdepth = grid_search.best_params_['max_depth']
        mfeature = grid_search.best_params_['max_features']
        msample = grid_search.best_params_['max_samples']
        minsampleleaf = grid_search.best_params_['min_samples_leaf']
    
        final_model = RandomForestRegressor(n_estimators = tree_number,max_depth=maxdepth,max_features=mfeature,max_samples = msample,
                                            min_samples_leaf=minsampleleaf)
    
    
    
        final_model.fit(normed_train_data, train_labels)
    
        output = final_model.predict(normed_train_data)

        output_test = final_model.predict(normed_test_data)
       


        #------------------------- making a dataframe -------------------------------------------------
        train_test = np.append(np.repeat("train",len(np.array(output))),np.repeat("test",len(np.array(output_test))))
        model_ozone = np.append(np.array(output),np.array(output_test))
        obs_ozone = np.append(np.array(train_labels),np.array(test_labels))
        inx = np.append(np.array(normed_train_data.index),np.array(normed_test_data.index)) # making a index, when we sample, the data is shuffled

        ozoneoutput = pd.DataFrame({'model_ozone':model_ozone,'obs_ozone':obs_ozone,"train_test":train_test,"inx":inx})



        ozoneoutput = ozoneoutput.sort_values(by=['inx'],ascending=True) # sort the index of the original data

        ozoneoutput.index = np.arange(0,len(ozoneoutput.index),1) # sort the index of the dataframe

        ozoneoutput['date'] = met_ozone_data_sub_noNAN['date'] # attach date
    
        ozoneoutput.to_csv(
            
            
                ozone_outputpath+str(tt)+"_"+region+"_"+str(year_for_test)+"_RFR_ozone_comparison.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
        
            
            
                )
    
    
        bestpara = pd.DataFrame(grid_search.best_params_.items())
        bestpara.columns = np.array(["para","value"])
    
    
        bestpara.to_csv(
            
            
                ozone_outputpath+str(tt)+"_"+region+"_"+str(year_for_test)+"_RFR_b.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
        
            
            
                )
    
    
        
        feature_list = train_dataset.columns.values
    
        importance = final_model.feature_importances_
    
        FItable = pd.DataFrame({'feature':feature_list,'score':importance})
    
        FItable.to_csv(
                ozone_outputpath+str(tt)+"_"+region+"testyear_"+str(year_for_test)+"_FItable.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
                )
    




        
    
    


    




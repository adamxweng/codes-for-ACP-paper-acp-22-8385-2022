import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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

allyears = np.array(np.arange(2015,2020,1)) # 2015 to 2019


# output path
ozone_outputpath = "/output/06_Ridge_grid/GridOut/"


# met data dir
metdir = "/meterology/06_Ridge_grid/metinput/"






for year_for_test in allyears:
    print(year_for_test)

    for tt in np.arange(0,len(regions_all),1):
    
        region = regions_all[tt] # grid ID
    
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

       
    
    
        test_dataset = met_ozone_data_sub_noNAN_nodatemonth.iloc[yearix,:]
    
        train_dataset = met_ozone_data_sub_noNAN_nodatemonth.drop(test_dataset.index)


        train_stats = train_dataset.describe()
        train_stats.head()
        
        ##!!!
        y_stats = train_stats['O3']
        y_stats = y_stats.transpose()

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


        

        normed_train_data.head()
        normed_test_data.head()
        
        ##!!!, for ridge regression, also standardize Y
        train_labels = (train_labels - y_stats['mean'])/y_stats['std']
        test_labels = (test_labels - y_stats['mean'])/y_stats['std']
        
        # -------------------------- gridSearchCV defining the matrix for hyperparameters ---------------
    
        
        
        alpha_i=np.arange(1,200,2) # 1 to 199
        
        
        
        param_grid = {
                'alpha': alpha_i,
                'fit_intercept': [True, False], # let model to choose whether to fit an intercept
                'max_iter':[1000],
                'random_state':[100]
             }


        #--------------------------- fitting the model ---------------------------------------------
        
    
        RidgeRegression = Ridge()
        
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
            
   
    
    
        
        
        grid_search = GridSearchCV(estimator = RidgeRegression, param_grid = param_grid, 
                          cv = trainvalidationSplit, n_jobs = -1, verbose = 2)
        
        # default using sklearn.metrics.r2_score
    

        grid_search.fit(normed_train_data, train_labels)
    
        grid_search.best_params_
        
        
    
        alpha_val = grid_search.best_params_['alpha']
    
        fitintercept = grid_search.best_params_['fit_intercept']
        
        rs = grid_search.best_params_['random_state']
        
        maxiter = grid_search.best_params_['max_iter']
    
        final_model = Ridge(alpha = alpha_val,fit_intercept = fitintercept,max_iter = maxiter,random_state = rs)
    
    
    
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
        
        # !!! need to set back to unstandardized
        
        ozoneoutput['model_ozone'] = (ozoneoutput['model_ozone'] * y_stats['std']) + y_stats['mean']
        ozoneoutput['obs_ozone'] = (ozoneoutput['obs_ozone'] * y_stats['std']) + y_stats['mean']
    
        ozoneoutput.to_csv(
            
            
                ozone_outputpath+str(tt)+"_"+region+"_"+str(year_for_test)+"_RidgeRegression_ozone_comparison.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
        
            
            
                )
    
    
        bestpara = pd.DataFrame(grid_search.best_params_.items())
        bestpara.columns = np.array(["para","value"])
    
    
        bestpara.to_csv(
            
            
                ozone_outputpath+str(tt)+"_"+region+"_"+str(year_for_test)+"_RidgeRegression_b.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
        
            
            
                )
    
    
        
        feature_list = train_dataset.columns.values
    
        coef_val = final_model.coef_
        intercept_val = final_model.intercept_
    
        coef_data = pd.DataFrame({'feature':feature_list,'Slope':coef_val,'intercept':intercept_val})
    
        coef_data.to_csv(
                ozone_outputpath+str(tt)+"_"+region+"testyear_"+str(year_for_test)+"_FItable.csv",
                header = True,
                sep = ",",index = False,encoding = 'UTF-8'
                )
    




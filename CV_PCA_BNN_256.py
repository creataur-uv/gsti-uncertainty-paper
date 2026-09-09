# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bnn (3.11.14.final.0)
#     language: python
#     name: python3
# ---

# %%
## imports
import helpers.raw_Data_Reader
import helpers.filter_Spectrometers_traits
import time
import sys
from sklearn.model_selection import train_test_split
import torch
import algos.GPR 
import gpytorch
import helpers.benchmark
import math
import time
import helpers.save_To_DB as db

# %%
## Select GPU

device = torch.device("cuda:0")  # use GPU 0

# %%
## import Data

full_Data = helpers.raw_Data_Reader.csv("./data/input")

# %%
import pandas

## List of traits
leaf_traits = [ 'LMA', 'Narea', 'Nmass', 'Parea', 'Pmass', 'LWC' ]

## List of columns for categories
cat_cols = list(['Phenological_stage','Plant_type','Soil'])

## List of Columns containing Reflectance Data
refl_cols = full_Data.filter(regex=r"^Wave_ ").columns.tolist()

## Dictionary to store traits related data
traits_DF = dict()

for trait in leaf_traits:
    traits_DF[trait] = dict()


## assign appropriate Spectrometers
# traits_DF['LMA']['Spectrometers'] = ['ASD FieldSpec 3', 'ASD FieldSpec 4', 'SE PSR+ 3500', 'SVC HR-1024i']
# traits_DF['Nmass']['Spectrometers'] = ['ASD FieldSpec 3', 'ASD FieldSpec 4', 'ASD FieldSpec 4 Hi-Res', 'SE PSR+ 3500', 'SVC HR-1024i'] 
# traits_DF['Narea']['Spectrometers'] = ['ASD FieldSpec 3', 'ASD FieldSpec 4', 'ASD FieldSpec 4 Hi-Res', 'SE PSR+ 3500', 'SVC HR-1024i'] 
traits_DF['LMA']['Spectrometers'] = ['ASD FieldSpec 4'] 
traits_DF['Nmass']['Spectrometers'] = ['ASD FieldSpec 4']
traits_DF['Narea']['Spectrometers'] = ['ASD FieldSpec 4'] 
traits_DF['Parea']['Spectrometers'] = ['ASD FieldSpec 3']
traits_DF['Pmass']['Spectrometers'] = ['ASD FieldSpec 3']
traits_DF['LWC']['Spectrometers'] = ['ASD FieldSpec 4'] 


## assign critical R2 for each trait
traits_DF['LMA']['min_R2'] =  0.6
traits_DF['Nmass']['min_R2'] = 0.6
traits_DF['Narea']['min_R2'] =  0.3
traits_DF['Parea']['min_R2'] = 0.0
traits_DF['Pmass']['min_R2'] = 0.0
traits_DF['LWC']['min_R2'] = 0.0

# %%
## create a list of wavelength filters for each trait

for trait in leaf_traits:
    traits_DF[trait]["wavelength_Filter"] = list()

# %%
import numpy
import algos.pca as pca
import algos.DeepGPR as dGP


import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

def cross_val_scores(
        X,
        y,
        num_Folds,
        min_R2,
        num_PC
        ):
    if len(X) != len(y):
        return "X and y must be of same length"
    
    idx = numpy.random.permutation(len(X))
    folds = numpy.array_split(idx, num_Folds)

    results = dict()

    for i in range(num_Folds):
        results[i] = dict()

    ## Counter for number of folds
    i = 0
    while i < num_Folds:
        test_X = X[folds[i]]
        test_y = y[folds[i]]

        train_X = numpy.delete(X, folds[i], axis=0)
        train_y = numpy.delete(y, folds[i], axis=0)


        ## Data Preprocessing

        # Rescaling of input parameters from -1 to 1
        # Separately for test_Data to avoid data leakage from
        mins_train_X, maxs_train_X = numpy.min(train_X, axis=0), numpy.max(train_X, axis=0)
        ranges_train_X = maxs_train_X - mins_train_X
        train_X = (2 * (train_X - mins_train_X) / ranges_train_X) - 1
        test_X = (2 * (test_X - mins_train_X) / ranges_train_X) - 1
        
        # Standardization of target
        target_mean = numpy.mean(train_y)
        target_std = numpy.std(train_y)
        train_y = (train_y - target_mean) / target_std

        # Model Training
        train_X, train_y, test_X, test_y = torch.tensor(train_X).to(device), torch.tensor(train_y).to(device), torch.tensor(test_X).to(device), torch.tensor(test_y).to(device)

        # PCA
        pc = pca.PCA_SVD(n_components=num_PC)

        train_X = pc.fit_transform(train_X)
        test_X = pc.transform(test_X)

        # initialize likelihood and model
        model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=num_PC, out_features=256),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=1),
            # nn.ReLU(),
            # bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=32, out_features=16),
            # nn.ReLU(),
            # bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=1),
            )

        model.cuda()
        model.double()
        train_y = train_y.unsqueeze(1)
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.01

        start = time.time_ns()

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for step in range(10000):
            pre = model(train_X)
            mse = mse_loss(pre, train_y)
            kl = kl_loss(model)
            cost = mse + kl_weight*kl
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        train_Time = time.time_ns() - start 

        start = time.time_ns()
        
        test_x = test_X.cpu()
        test_y = test_y.cpu()

        model = model.cpu()
        models_result = numpy.array([model(test_x).data.numpy() for k in range(10000)])
        models_result = models_result[:,:,0]    
        models_result = models_result.T
        mean_values = numpy.array([models_result[i].mean() for i in range(len(models_result))])
        std_values = numpy.array([models_result[i].std() for i in range(len(models_result))])
        
        mean_pred = mean_values
        stddev_pred = std_values
        
        
        # mean_pred, stddev_pred = model.infer(test_x=test_X.cuda(), test_y=test_y.cuda())

        infer_Time = time.time_ns() - start


        

        ## Rescaling and benchmarking 
        observed, predicted, predicted_stddev = test_y, mean_pred, stddev_pred

        predicted = predicted * target_std + target_mean
        predicted_stddev = predicted_stddev * target_std


        result = helpers.benchmark.benchmark(observed, predicted)
        # print(result['R2'])

        ## re-run with same fold if model suffers from local minima effects
        # if result["R2"] < min_R2:
        #     continue

        # print(f"\n{i}, {trait} ", end=", ")
        # for key, value in result.items():
        #     print(f"{key}: {value:8.6f}", end=", ")


        results[i]['benchmarks'] = result 
        # results[i]['length_Scales'] = model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0].flatten()

        i = i + 1
        # print(f"train_X: {train_X.shape} \t train_y: {train_y.shape} \t test_X: {test_X.shape} \t test_y: {test_y.shape} \t ")
        # print(fold)

    return results


# %%
## Main Loop
import pysqlite3 as sqlite3
import pickle



for trait in leaf_traits:
# for trait=leaf_traits[0]:

    # if trait != leaf_traits[0]:
    #     continue
    for num_PC in range(1, 50):
        num_Folds = 5

        
        ## Filter Reflectance Columns
        filtered_refl_cols = refl_cols
        ## Filter Trait Specific Data
        in_Data = helpers.filter_Spectrometers_traits.filter(in_Data=full_Data, spectrometers=traits_DF[trait]['Spectrometers'], traits=cat_cols + [trait])
        
        ## Data preprocessing
        # filter the outliers based on target variable
        target_mean = in_Data[trait].mean()
        target_std = in_Data[trait].std()
        in_Data = in_Data[(in_Data[trait] >= target_mean - 2.5 * target_std) & (in_Data[trait] <= target_mean + 2.5 * target_std)]

        in_X = in_Data[filtered_refl_cols].to_numpy()
        in_y = in_Data[trait].to_numpy()



        ## Cross-Validition

        cross_val_results = cross_val_scores(
            X=in_X,
            y=in_y,
            num_Folds=num_Folds,
            min_R2=traits_DF[trait]['min_R2'],
            num_PC=num_PC
        )

        wavelengths_for_trait = set()
        results = list()
        for i in range(num_Folds):
            print(cross_val_results[i]['benchmarks']['R2'], end=", ")
            results.append(cross_val_results[i]['benchmarks'])
            # wavelengths_for_trait = wavelengths_for_trait.union(set([filtered_refl_cols[numpy.argmax(cross_val_results[i]['length_Scales'])]]))

        wavelengths_for_trait = {str(x) for x in wavelengths_for_trait}

        traits_DF[trait]['wavelength_Filter'] = traits_DF[trait]['wavelength_Filter'] + list(wavelengths_for_trait)

        # ## Save to Database
        to_DB = dict()
        to_DB["cross_val_results"] = results
        to_DB["num_wavelengths"] = len(filtered_refl_cols)
        # to_DB["wavelengths_used"] = filtered_refl_cols
        to_DB["num_obs"] = len(in_Data)
        to_DB["trait"] = trait
        to_DB["mlra"] = "CV_PCA_BNN_256"

        # # Data to Data Base
        print("Saving to Data Base")
        DB_Name = f"data/output/CV_PCA_BNN_256.db"
        table_Name = "simulations"

        myDB = db.db(DB_Name, table_Name)

        myDB.insert(f"""{to_DB["mlra"]}_{trait}_{to_DB['num_wavelengths']}""", to_DB)
        myDB.closeDB()


        print(f"Trait: {trait}, Num_Obs: {len(in_Data)}, Num_Wavelengths: {len(filtered_refl_cols)}, num_PC: {num_PC}")

# %%

# %%

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bnn (3.11.14)
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


def cross_val_scores(
        X,
        y,
        num_Folds,
        min_R2
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

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_X.size(-1)))
        model = algos.GPR.GPR(train_X, train_y, likelihood,kernel)
        model = model.to(device)
        model = model.double()

        start = time.time_ns()
        model.optimize(training_iter=1000)
        train_Time = time.time_ns() - start 

        start = time.time_ns()
        predictions = model.infer(X=test_X)
        infer_Time = time.time_ns() - start

        mean_pred = predictions.mean
        stddev_pred = predictions.stddev

        ## Rescaling and benchmarking 
        observed, predicted, predicted_stddev = test_y.cpu(), mean_pred.cpu(), stddev_pred.cpu()

        predicted = predicted * target_std + target_mean
        predicted_stddev = predicted_stddev * target_std


        result = helpers.benchmark.benchmark_new(observed.numpy(), predicted.numpy(), predicted_stddev.numpy())

        ## re-run with same fold if model suffers from local minima effects
        # if result["R2"] < min_R2:
        #     continue

        # print(f"\n{i}, {trait} ", end=", ")
        # for key, value in result.items():
        #     print(f"{key}: {value:8.6f}", end=", ")


        results[i]['benchmarks'] = result 
        results[i]['length_Scales'] = model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0].flatten()

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

    while len(traits_DF[trait]['wavelength_Filter']) < 2150:
        num_Folds = 5

        
        ## Filter Reflectance Columns
        filtered_refl_cols = numpy.array(sorted(list(set(refl_cols) - set(traits_DF[trait]['wavelength_Filter']))))
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
            min_R2=traits_DF[trait]['min_R2']
        )

        wavelengths_for_trait = set()
        results = list()
        for i in range(num_Folds):
            print(cross_val_results[i]['benchmarks']['R2'], end=", ")
            results.append(cross_val_results[i]['benchmarks'])
            wavelengths_for_trait = wavelengths_for_trait.union(set([filtered_refl_cols[numpy.argmax(cross_val_results[i]['length_Scales'])]]))

        wavelengths_for_trait = {str(x) for x in wavelengths_for_trait}

        traits_DF[trait]['wavelength_Filter'] = traits_DF[trait]['wavelength_Filter'] + list(wavelengths_for_trait)

        ## Save to Database
        to_DB = dict()
        to_DB["cross_val_results"] = results
        to_DB["num_wavelengths"] = len(filtered_refl_cols)
        to_DB["wavelengths_used"] = filtered_refl_cols
        to_DB["num_obs"] = len(in_Data)
        to_DB["trait"] = trait
        to_DB["mlra"] = "CV_GPR_ARD_SE"

        # Data to Data Base
        print("Saving to Data Base")
        DB_Name = f"data/output/CV_GPR_ARD_SE.db"
        table_Name = "simulations"

        myDB = db.db(DB_Name, table_Name)

        myDB.insert(f"""{to_DB["mlra"]}_{trait}_{to_DB['num_wavelengths']}""", to_DB)
        myDB.closeDB()


        print(f"Trait: {trait}, Num_Obs: {len(in_Data)}, Num_Wavelengths: {len(filtered_refl_cols)}")
# CODARFE Class (python3.10.12)

* [Installing dependences](#Installing-dependences)
* [Create a CODARFE instance or load one](#Create-a-CODARFE-instance-or-load-one)
* [Create the model](#Create-the-model)
* [Save the Instance](#Save-the-Instance)
* [Creating graphics](#Creating-graphics)
  - [Correlation Plot](#Correlation-Plot)
  - [Hold Out Validation](#Hold-Out-Validation)
  - [Relevant Predictor Plot](#Relevant-Predictor-Plot)
  - [Heat Map](#Heat-Map)
* [Predicting target variable for new samples](#Predicting-target-variable-for-new-samples)
* [Model Atributes](#Model-Atributes)


## Installing dependences
*It is recommended that you build a virtual environment using Python 3.10.12 for this.*  

After downloading the requirements.txt file, run the following command:
```sh
pip install -r requirements.txt
```

### 1) Create a CODARFE instance or load one

Create the CODARFE instance by passing to it:
  1. The path to the **predictor** table (count table)  
    - It can be the following format:
       - .csv (separated by comma)
       - .tsv (separated by tab)
       - .biom (**Bi**ological **O**bservation **M**atrix format)
       - .qza (qiime2 format compact)
  2. The path to the **metadata** table (where your target variable is)  
    - It can be the following format:
       -  .csv (separated by comma)
       -  .tsv (separated by tab)
  3. The name of the **target** variable as it is in the metadata table

```python
 coda = CODARFE(path2Data       = <path_to_predictor_table>,  
                path2MetaData   = <path_to_metadata_table>,  
                metaData_Target = <name_of_target_variable_in_matadata>)  
```

***OR***

If you already created an CODARFE instance and want to load it, you can just create an empty instance and load the file instance:

```python
# Create a empty instance
coda = CODARFE()
# Then you can call the Load_Instance method
coda.Load_Instance(path2instance = <path_to_file_instance.foda>)
# The file is the one with the .foda extension 
```

### 2) Create the model

You can create the model with all the default parameters by running:
```python
coda.CreateModel()
```

If you want to *change the model configuration*, this are all the parameters:
```python
coda.CreateModel( write_results            = True,
                  path_out                 = '',
                  name_append              = '',
                  rLowVar                  = True,
                  applyAbunRel             = True,
                  percentage_cols_2_remove = 1,
                  n_Kfold_CV               = 10,
                  weightR2                 = 1.0,
                  weightProbF              = 0.5,
                  weightBIC                = 1.0,
                  weightRMSE               = 1.5,
                  n_max_iter_huber         = 100
                )
```
* *write_results:* Defines if the results will be written. The results include the selected predictors and the metrics for its selection.
* *path_out:* Where to write the results
* *name_append:* The name to append in the end of the file with the results
* *rLowVar:* Flag to define if it is necessary to apply the removal of predictors with low variance. Set as False if less than 300 predictors.
* *applyAbunRel:* Flag to define if it is necessary to apply the relative abundance transformation. Set as False if the data is already transformed
* *percentage_cols_2_remove:* Percentage of the total predictors removed in each iteraction of the RFE. HIGH IMPACT in the final result and computational time.
* *n_Kfold_CV:* Number of folds in the Cross-validation step for the RMSE calculation. HIGH IMPACT in the final result and computational time.
* *weightR2:* Weight of the R² metric in the model’s final score
* *weightProbF:* Weight of the Probability of the F-test metric in the model’s final score
* *weightBIC:* Weight of the BIC metric in the model’s final score
* *weightRMSE:* Weight of the RMSE metric in the model’s final score
* *n_max_iter_huber:* Maximum number of iterations of the huber regression. HIGH IMPACT in the final result and computational time.
  
### 3) Save the Instance
Just save the instance with the filename you want.  
If no filename is provided it will save in the *same directory as the metadata* with the name of 'CODARFE_MODEL.foda'

```python
coda.Save_Instance(path_out    = <path_to_folder>,
                   name_append = <name>)
```
* *path_out:* Path to folder where it will be saved. If no path is provided it will save in the *same directory as the metadata* with the name of 'CODARFE_MODEL.png'
* *name_append:* Name to concatenate in the final filename.

## Creating graphics

### Correlation Plot

Display the correlation between the real target variable and the prediction of the own data.
```python
coda.Plot_Correlation(path_out    = <path_to_folder>,
                      name_append = <name>)
```
* *path_out:* Path to folder where it will be saved. If no path is provided it will save in the *same directory as the metadata* with the name of 'Correlation.png'
* *name_append:* Name to concatenate in the final filename.
### Hold Out Validation

Display a box plot of the Hold out validation's mean absolute error.
```python
coda.Plot_HoldOut_Validation( n_repetitions = 100,
                              test_size     = 20,
                              path_out      = <path_to_folder>,
                              name_append   = <name>)
```
* *n_repetitions:* Number of times will be performed an Hold-out.  (Number of dots in the final image)
* *test_size:* Percentage of the total data that will be used as test during the Hold-out.
* *path_out:* Filename of the output.If no filename is provided it will save in the *same directory as the metadata* with the name of 'HoldOut_Validation.png'
* *name_append:* Name to concatenate in the final filename. (Use it to differentiate plots with different parameters)

### Relevant Predictors Plot

Display the most relevant predictors selected by CODARFE and its strength and direction of correlation.
```python
coda.Plot_Relevant_Predictors(n_max_features = 100,
                              path_out       = <path_to_folder>,
                              name_append    = <name>)
```
* *n_max_features:* Maximum number of predictors to be displayed. (Will select the most important ones)
* *path_out:* Filename of the output. If no filename is provided it will save in the *same directory as the metadata* with the name of 'HoldOut_Validation.png'
* *name_append:* Name to concatenate in the final filename. (Use it to differentiate plots with different parameters)
### Heat Map

Display a Heat map with the Center-Log-Ratio abundance of each predictor in relation to the target variable. The target variable is  sorted from largest to smallest from left to right.
```python
coda.Plot_Heatmap(path_out    = <path_to_folder>,
                  name_append = <name>)
```
* *path_out:* Filename of the output. If no filename is provided it will save in the *same directory as the metadata* with the name of 'HeatMap.png'
* *name_append:* Name to concatenate in the final filename. (Use it to differentiate plots with different parameters)
## Predicting target variable for new samples

All you need to do is to call the predict function and pass to it the path to the data and if it needs to be transformed in relative abundance.
```python
coda.Predict(path2newdata = <path_to_new_data>,
             applyAbunRel = True,
             writeResults = True,
             path_out     = <path_out>
             name_append  = <name>)
```
* *path2newdata:* The path to the new data
    - It can be the following format:
       - .csv (separated by comma)
       - .tsv (separated by tab)
       - .biom (qiime2 format)
       - .qza (qiime2 format compact)
* *applyAbunRel:* Flag to apply relative abundance transformation
* *writeResults:* Flag to write the results
* *path_out:* Filename of the output. If no filename is provided it will save in the *same directory as the metadata* with the name of  'Prediction.csv'
* *name_append:* Name to concatenate in the final filename. (Use it to differentiate predictions from the same model)

### Model Atributes

*  *results:* **Dictionary** containing all the metrics values with the best result.
*  *score_best_model:*  **Float** value with the minmax result from all the 4 metrics for the best model. (selected after the RFE)
*  *selected_taxa:* **List** of selected predictors.
*  *weights:* **Pandas Dataframe** with predictors names and coeficient weights
  
```python
results = coda.results
score_best_model = coda.score_best_model
selected_taxa = coda.selected_taxa
weights = coda.weights
```



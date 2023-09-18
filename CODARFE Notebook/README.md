# CODARFE Notebook (google colab version)
This version is dedicated for users with low level of programming. You can simply download the CODARFE.ipynb file and upload it to your google drive account. Than you just need to open it as a Google Colab document and it is ready to go.

## How to use it

After you open the file in your google drive or any other notebook enviroment just follow the steps bellow:

### Create a CODARFE instance or load one

Create the CODARFE instance by passing to it:
  1. The path to the **predictor** table (count table)  
    - It can be the following format:
       - .csv (separated by comma)
       - .tsv (separated by tab)
       - .biom (qiime2 format)
       - .qza (qiime2 format compact)
  2. The path to the **metadata** table (where your target variable is)
    - It can be the following format:
       -  .csv (separated by comma)
       -  .tsv (separated by tab)
  3. The name of the **target** variable as it is in the metadata table

Example:  
```python
 coda = CODARFE(path2Data       = <path_to_predictor_table>,  
                path2MetaData   = <path_to_metadata_table>,  
                metaData_Target = <name_of_target_variable_in_matadata>)  
```

***OR***

If you already created an CODARFE instance and want to load it, you can just create an empty instance and load the file instace:

```python
coda = CODARFE()
# Then you can call the LoadInstance method
```

### Create the model

You can create the model with all the default parameter by running:
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
* *write_results:* Defines if the results will be written. The results includes the selected predictos and the metrics for its selection.
* *path_out:* Where to write the results
* *name_append:* The name of the file with the results
* *rLowVar:* Flag to define if is necessary to apply the removal of predictors with low variance. Set as False if less then 300 predictors.
* *applyAbunRel:* Flag to define if is necessary to apply the relative abundace transformation. Set as False if the data is already transformed
* *percentage_cols_2_remove:* Percentage of the total predictors removed in each iteraction of the RFE. HIGH IMPACT in the final result and computational time.
* *n_Kfold_CV:* Number of folds in the Cross-validation step for the RMSE calculation. HIGH IMPACT in the final result and computational time.
* *weightR2:* Weight of the R² metric in the model’s final score
* *weightProbF:* Weight of the Probabilit of the F-test metric in the model’s final score
* *weightBIC:* Weight of the BIC metric in the model’s final score
* *weightRMSE:* Weight of the RMSE metric in the model’s final score
* *n_max_iter_huber:* Maximum number of iteration of the huber resgressor. HIGH IMPACT in the final result and computational time.

# CODARFE CLASS

## Installing dependences
*It is recommended that you build a virtual environment using Python 3.10.12 for this stage.*  

After downloading the requirements.txt file, run the following command:
```sh
pip install -r requirements.txt
```

## Usage


### Creating model *From scracth*
```python
from CODARFE_class import CODARFE

coda = CODARFE(path2Data=<path_2_count_table>,
               path2MetaData=<path_2_metadata>,
               metaData_Target=<Target_name_in_metadata>)

coda.CreateModel()
```
*All models parameters (default values)*
```python
CreateModel(write_results: bool =True,
            path_out: str ='',
            name_append: str ='',
            rLowVar: bool =True,
            applyAbunRel: bool = True,
            percentage_cols_2_remove: int =1,
            n_Kfold_CV: int=10,
            weightR2: int =1.0,
            weightProbF: float=0.5,
            weightBIC: float=1.0,
            weightRMSE: float=1.5,
            n_max_iter_huber: int=100):
```
* **write_results**: Flag for writting the results.
* **path_out**: The path on where the results will be written
* **rLowVar**: Apply the low variance columns elimination. (For small datasets (less than 300 columns), set as False).
* **applyAbunRel**: Apply the transformation for relative abundance. (Set as False if your data is already in relative abundance).
* **percentage_cols_2_remove**: The number of columns to eliminate in each RFE iteration (for example, use 1 for 1%).
* **n_Kfold_CV**: In the Cross-validation step, the number of folds.
* **n_max_iter_huber**: The Huber regression\'s maximum number of iterations.
* **weightR2**: Alter the R2\'s default weight in the RFE score.
* **weightProbF**: Alter the Prob(F)\'s default weight in the RFE score.
* **weightRMSE**: Alter the RFE score\'s default weighting of the RMSE.
* **weightBIC**: Alter the default weight of the BIC in the RFE score.  

### *Loading existing model*
```python
coda.Load_Instance(<CODARFE_model.foda>)
```

### *Predicting* new samples

```python
coda.Predict(<path_2_new_samples>)
```

### Plotting

*Correlation learned Vs Real*

```python
coda.Plot_Correlation()
```

*Hold Out Validation*

```python
coda.Plot_HoldOut_Validation(n_repetitions = 100,
                             test_size=0.2)
```
* **n_repetitions**: The number of Hold-Out repetitions (represented by the number of dots in the box plot).
* **test_size**: The percentage of the dataset that is utilized in the test (for example, 20% uses 20).
*Relevant predictors*

```python
coda.Plot_Relevant_Predictors(n_max_features=100)
```
* **n_max_features**: The maximum number of predictors to display.
*Heat Map*

```python
coda.Plot_Heatmap()
```





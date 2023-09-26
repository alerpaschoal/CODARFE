# CODARFE script for CMD call

## Installing dependences
*It is recommended that you build a virtual environment using Python 3.10.12 for this stage.*  

After downloading the requirements.txt file, run the following command:
```sh
pip install -r requirements.txt
```

## How to run the program

*Basic:*
```
python3 CODARFE_scrit.py -d <path_2_count_table> -m <path_2_metadata> -t <Target_name_in_metadata> -o <path_out>
```

*Loading instance:*
```shell
python3 CODARFE_scrit.py -l <CODARFE_instance.foda> **[others parameters]
```

*Predicting:*
```shell
python3 CODARFE_scrit.py -l <CODARFE_instance.foda> -p <path_2_new_samples>
```

**ALL PARAMETERS:**
* **-h --Help**: Display this Help.  
* **-d --Data**: The name of the counting table data file.  
* **-m --Metadata**: The name of the target\'s metadata file.  
* **-t --Target**: The name of the metadata column that the Target variable corresponds to.  
* **-o --Output**: The FOLDER\'s path, where ALL files will be saved.  
* **-na --NameAppend**: To add a name to the model and results filenames.  
* **-p --Predict**: The name of the data file containing new samples that the model will predict.
  * **-po --PredictionOutput**: The path that is used when writing the prediction.
  * **-pna --PredictionNameAppend**: The name that is added at the end of the predictions filename.
  * **-pra --ApplyRAforPrediction**: Apply the transformation for relative abundance to the new samples. (Set as False/f/F if your data is already in relative abundance).
* **-l --LoadInstance**: The name of the instance file for an existing CODARFE model (.foda file)
  * Since it is saved in the Instace file, it is not necessary to use -d, -m, or -t when using -l.  

*MODEL PARAMETERS*  

* **-rlv --RemoveLowVar**: Apply the low variance columns elimination. (For small datasets (less than 300 columns), set as False/f/F).
* **-ra --RelativeAbundance**: Apply the transformation for relative abundance. (Set as False/f/F if your data is already in relative abundance).
* **-cr --PercentageColsToRemove**: The number of columns to eliminate in each RFE iteration (for example, use 1 for 1%).
* **-nf --NumberOfFolds**: In the Cross-validation step, the number of folds.
* **-hb --HuberRegression**: The Huber regression\'s maximum number of iterations.
* **-r2 --R2weight**: Alter the R2\'s default weight in the RFE score.
* **-pf --ProbFweight**: Alter the Prob(F)\'s default weight in the RFE score.
* **-rmse --RMSEweight**: Alter the RFE score\'s default weighting of the RMSE.
* **-bic --BICweight**: Alter the default weight of the BIC in the RFE score.  

*PLOTS*  

* **-pc --PlotCorrelation**: Create a correlation plot with the generalized Vs target and save it.
  * **-pcna --PlotCorrelationNameAppend**: The name to put to the end of the plot\'s filename..
* **-ho --PlotHoldOut**: Create and save the Hold-Out validation box-plot with the MAE.
  * **-horep --HoldOutRepetitions**: The number of Hold-Out repetitions (represented by the number of dots in the box plot).
  * **-hots --HoldOutTestSize**: The percentage of the dataset that is utilized in the test (for example, 20% uses 20).
  * **-hona --HoldOutNameAppend**: The name to put to the end of the plot\'s filename.
* **-hm --PlotHeatMap**: Create and save a Heat-Map of the selected predictors\' CLR transformed abundance.
  * **-hmna --PlotHeatMapNameAppend**: The name to put to the end of the plot\'s filename.
* **-rp --PlotRelevantPredictors**: Create and save a bar plot containing the top -rpmax predictors, as well as their strengths and directions of correlation to the target.
  * **-rpmax --RelevantPredictorMaximum**: In the -rp, pick the maximum number of predictors to display.
  * **-rpna --RelevantPredictorNameAppend**: The name to put to the end of the plot\'s filename.
# CODARFE
Here we present the CODARFE: **CO**mpositional **D**ata **A**nlises with **R**ecursive **F**eature **E**limination  

CODARFE is a tool for selecting predictors (such as bacteria) for any continuous environmental variable (such as pH, illness measurements, diseases index, etc).  

For predictors selection, it combines Recursive Feature Elimination in conjunction with Cross-validation and four sophisticated and strong statistical measures, returning the most important predictors associated with the target variable.  

CODARFE's main distinguishing feature is its capacity to **PREDICT** the target variable in new samples based on past data. It is accomplished via an imputation module that employs correlation in conjunction with compositional transformation to solve the missing predictor problem and predict the target variable using a Random Forest model created through the predictors selected by the “predictors selection” step.  


![fluxoCODARFEsimples](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/e3370d81-fa63-42b0-b168-9e0b3e7bdb0e)


## INPUT - What are the inputs?
Codarfe needs only **3 inputs**:
* **1)** The path for the file where you've saved the microbiome data (counting data, feature table).  
  - This data must contain only the counting data.
  - It can be in the formats QZA, BIOM, CSV, or TSV. Although Excel is not supported, it is simple to export data as a csv file, which is supported.
  - All metadata (such as the target or any other) MUST be in a separated file.
  - The rows MUST be the samples and the columns the predictors (e.g.: OTU, ASVs, phylogeny ,etc).
  - Each sample must have a unique identification.
    - The formats qza and biom already have an identifier by default.
    - The formats csv or tsv, the table's first column must contain the identifier.    

* **2)** The path of the file in which you have the metadata.  
  - The target variable MUST be included in this metadata.  
  - It can be in the formats CSV, or TSV. Although Excel is not supported, it is simple to export data as a csv file, which is supported.
  - Each sample MUST have an identifier, which must be the first column in the table.
  - The microbiome data table (counting data, feature table) and metadata`s identifiers must match.
    
* **3)** The name of the metadata column corresponding to the target variable.
  - The name MUST match the column name of interest  

## OUTPUT - What are the outputs?
* **1)** Correlation Plot
  - The correlation between the predicted and the real (train dataset)
* **2)** Box-plot Hold-out validation
  - The Mean Absolute Error (MAE) using a hold-out validation (train dataset) 100 times (configurable)
* **3)** Relevant predictors Plot
  - The strength of the correlation of each selected predictor
* **4)** Heat-map
  - A heatmap showing the abundance of each selected predictor relative to the target distribution (highest to lowest, left to right)
* **5)** Selected predictors
  - A list with all the selected predictors
* **6)** Scores (train dataset only)
  - $R^2$ adjusted
  - Rooted Mean Squared Error (RMSE) in a 10-fold cross-validation (configurable)
  - $\rho_{value}$ of the selected predictors (< 0.5 means they are significant)
  - Score: The minmax-sum of the previous ones
   
## Some limitations and considerations:
  * The data MUST be compositional (e.g. count table).
  * The target variable MUST be continuous.
  * For prediction, additional samples SHOULD be gathered in the same manner as the data used for training (for further information, read the Results and Discussion sections from the original article)
  * We recommend not using the “low variance removal” (“r_low_var = TRUE”) for databases with a small number of predictors (e.g., less than 500).
  * Do not apply the relative abundance transformation if your database is already in relative abundance.
      - The python version used was 'Python 3.10.12' 

## It cames with 4 formats  for distinc uses
  * CODARFE for Windows users
  * CODARFE notebook (google colab format)
  * CODARFE class for pipeline integration
  * CODARFE python script for CMD call

### which format should I use?

* If you just want an easy and fast use of the tool in an windows environment without the need to code a single line, try “CODARFE for Windows”. It is a graphical interface for the CODARFE tool.
* If you know the basics of programming and is used to jupyter environment or google colaboratory environment use “CODARFE Notebook”. (Can be used directly in Google colaboratory).
* If you want to incorporate CODARFE into your own pipeline by using a single class and its functionalities, use the "CODARFE class".
* If you want to utilize CODARFE in a CMD call, use the "CODARFE python script for CMD call". Check the tutorial inside the folder "CODARFE python script for CMD call" for the commands instructions.

Inside each folder you will find a tutorial with examples of how to run CODARFE.

# Bugs and errors?
Because most of the packages are constantly updated, it is possible that it could affect the tool versions for non-Windows users. If you find any bugs or strange behavior, please report it to us through the email: murilobarbosa@alunos.utfpr.edu.br  

# Citation
If you use CODARFE for your research, please, don't forget to cite us throught [citation]

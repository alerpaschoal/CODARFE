# CODARFE
Here we present the CODARFE: **CO**mpositional **D**ata **A**nlises with **R**ecursive **F**eature **E**limination  

CODARFE is a tool for selecting predictors (such as bacteria) for *any* continuous environmental variable (such as pH, illness measurements, and so on).  

For feature selection, it combines Recursive Feature Elimination in conjunction with Cross-validation and four sophisticated and strong statistical measures, returning the most important predictors associated with the target variable.  

CODARFE's main distinguishing feature is its capacity to **PREDICT** the target variable in new samples based on past data. It is accomplished via an imputation module that employs correlation in conjunction with compositional transformation.  


![fluxoCODARFEsimples](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/e3370d81-fa63-42b0-b168-9e0b3e7bdb0e)


## What are the inputs?
Codarfe needs only **3 inputs**:
* **1)** The path of the file where you've saved the counting data.  
  - This counting data often comes from a 16-sample genome database or from whatever else you want to link to an ambient continuous variable.   
  - It can be in the forms QZA, BIOM, CSV, or TSV. Although Excel is not supported, it is simple to export data as a csv file, which is supported.  
  - Each sample must have a unique identification. If the format is csv or tsv, you must determine whether it has an identifier. The formats qza and biom already handle this.  
    * In this scenario, the table's first column must contain the identity.  

* **2)** The path of the file in which you have the metadata.  
  - The variable you want to analyze must be included in this metadata.  
  - Each sample must also have an identification, which must be the first column in the table.  The counting table and this identifier must match.
* **3)** The name of the metadata column that belongs to the variable you want to analyze.  
   
## Some limitations and considerations:
  * The data MUST be compositional (e.g. count table).
  * The target variable MUST be continuous.
  * For prediction, additional samples SHOULD be gathered in the same manner as the data used for training (for further information, read the Results section of the original article).
  * Do not use the low variance removal for databases with a small number of predictors (e.g., less than 300).
  * Do not apply the relative abundance transformation if your database is already in relative abundance (e.g., not absolute counts).
      - The python version used was 'Python 3.10.12' 

## It cames with 4 formats  for distinc uses
  * CODARFE for Windows users
  * CODARFE notebook (google colab format)
  * CODARFE class for pipeline integration
  * CODARFE python script for CMD call

### which format should I use?

* If you are a Windows user or don't know the basics of programming, try "CODARFE for Windows"
* If you know the basics of programming and wish to use the tool with small datasets (less than 100k predictors and 500 samples), use the notebook version.
* If you are a programmer and want to incorporate CODARFE into your own pipeline by using a single class and its functionalities, use the "CODARFE class."
* If you have a large dataset or want to utilize CODARFE in a virtual server, use the "CODARFE python script for CMD call."


Inside each folder you will find a tutorial with examples of how to run the CODARFE.

# Bugs and errors?
Because it was done in Python and most of the packages are constantly updated, it is possible that it could affect the tool versions for non-Windows users.
If you find any bugs or strange behavior, please report us at the email: murilobarbosa@alunos.utfpr.edu.br  

# Citation
If you use CODARFE for your research, please, don't forget to cite us throught [citation]

# CODARFE
Here we present the CODARFE: **CO**mpositional **D**ata **A**nlises with **R**ecursive **F**eature **E**limination  

CODARFE is a tool for selecting predictors (such as bacteria) for any continuous environmental variable (such as pH, illness measurements, diseases index, etc).  

For predictors selection, it combines Recursive Feature Elimination in conjunction with Cross-validation and four sophisticated and strong statistical measures, returning the most important predictors associated with the target variable.  

CODARFE's main distinguishing feature is its capacity to **PREDICT** the target variable in new samples based on past data. It is accomplished via an imputation module that employs correlation in conjunction with compositional transformation to solve the missing predictor problem and predict the target variable using a Random Forest model created through the predictors selected by the “predictors selection” step.  

![github](https://github.com/user-attachments/assets/4013a648-882f-4bde-8592-c3093a811293)




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
  * Since the tool assumes the target is positive, any negative numbers will cause the target to shift (the predictions will be returned without the shift).
  * For prediction, additional samples SHOULD be gathered in the same manner as the data used for training (for further information, read the Results and Discussion sections from the original article)
  * We recommend not using the “low variance removal” (“r_low_var = TRUE”) for databases with a small number of predictors (e.g., less than 500).
  * Do not apply the relative abundance transformation if your database is already in relative abundance.
      - The python version used was 'Python 3.10.12' 

## It cames with 5 formats  for distinc uses
  * CODARFE for Windows users
  * CODARFE notebook (google colab format)
  * CODARFE class for pipeline integration
  * CODARFE python script for CMD call
  * CODARFE on [MGnify](https://shiny-portal.embl.de/shinyapps/app/06_mgnify-notebook-lab?jlpath=mgnify-examples/home.ipynb)

### Which format should I use?

* If you just want an easy and fast use of the tool in an windows environment without the need to code a single line, try “CODARFE for Windows”. It is a graphical interface for the CODARFE tool.
* If you know the basics of programming and is used to jupyter environment or google colaboratory environment use “CODARFE Notebook”. (Can be used directly in Google colaboratory).
* If you want to incorporate CODARFE into your own pipeline by using a single class and its functionalities, use the "CODARFE class".
* If you want to utilize CODARFE in a CMD call, use the "CODARFE python script for CMD call". Check the tutorial inside the folder "CODARFE python script for CMD call" for the commands instructions.
* If you want to analyze projects available on MGnify, use the Mgnify version for quick access and download of any project available on MGnify for version 4.1 or 5.

Inside each folder you will find a tutorial with examples of how to run CODARFE.

# CODARFE was not able to generalize my data! What can I do?
If you got the error about the data generalization or the $R^2$ is too low or the $\rho_{value}$ is not significant, don't panic (*yet*), there are some tricks you can try:

* **1)** If you have less than 1k columns (you can push a little further if you have time) you can try setting the flag **rLowVar** to **False**. This will make the tool examine every single column. It will also make the tool run much slower, but it may find a better result, since it is looking more "carefully". (During the creation of the tool, we tested on a dataset with over 10k columns and 1k samples and it took a few hours to finish)
* **2)** You can choose not to allow the target transformation (**allow_transform_high_variation** = **False**). In most cases, this transformation will help the model, but in some special cases, the raw target may yield a better result.
* **3)** You can try to remove fewer predictors on each step of the RFE (**percentage_cols_2_remove** = **0.1**). This will make the tool run much slower, but like option 1, it will take a better look at the predictors.
* **4)** You can increase the maximum iterations of the Huber regressor (**n_max_iter_huber** = **1000**). This may help during the predictor selection step, allowing the model to create a better fit on the data.
* **5)** You can also combine multiple suggestions above. Just remember that combining options *1* and *3* may cause a **severe increase in runtime**.

If after testing all the possibilities above the model is still yielding bad results, check if your target is appropriate for the tool. For example, if your target is between 0 and 1, it means that it is not appropriate for the methods used in this tool, and in this specific case, you may want to try applying a **logit** to your target **before** using the tool.

# Bugs and errors?
Because most of the packages are constantly updated, it is possible that it could affect the tool versions for non-Windows users. If you find any bugs or strange behavior, please report it to us through the email: murilobarbosa@alunos.utfpr.edu.br & paschoal@utfpr.edu.br 

# Citation Requirement for Using CODARFE

Thank you for using **CODARFE**. If you find this tool useful for your research or any other work, we kindly request that you cite our article in your publications and presentations. Proper citation is essential for acknowledging the effort involved in developing and maintaining this tool.

**Citation Information:**

*Article Title:* **CODARFE: Unlocking the prediction of continuous environmental variables based on microbiome**  

*Authors:* Murilo Caminotto Barbosa \*,  Joao Fernando Marques da Silva, Leonardo Cardoso Alves, Robert D. Finn,  Alexandre R Paschoal \*

*DOI:* https://doi.org/10.1101/2024.07.18.604052

Corresponding authors e-mail: murilobarbosa@alunos.utfpr.edu.br & paschoal@utfpr.edu.br

BibTeX:

~~~
@article{barbosa2024codarfe,
  title={CODARFE: Unlocking the prediction of continuous environmental variables based on microbiome},
  author={Barbosa, Murilo Caminotto and da Silva, Joao Fernando Marques and Alves, Leonardo Cardoso and Finn, Robert D and Paschoal, Alexandre R},
  journal={bioRxiv},
  pages={2024--07},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
~~~

**Important Note:**

By using **CODARFE**, you agree to cite the above-mentioned article in any publication, presentation, or other work that makes use of this tool. Proper citation helps us to track the impact and usage of our work, and supports the continued development and maintenance of the tool.

Thank you for your cooperation and support.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.


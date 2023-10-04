# CODARFE for Windows users (Visual interface)

The Windows operating system was used in the development of this version of CODARFE, which features a graphical user interface. For those who have had little to no experience with programming, this version is ideal. For each analysis you can perform using this interface, go to the instruction below.

* [Creating a model from scratch](#Creating-a-model-from-scratch)
* [Ploting the results](#Ploting-the-results)
  - [Correlation Plot](#Correlation-Plot)
  - [Hold Out Validation Plot](#Hold-Out-Validation-Plot)
  - [Heat map Plot](#Heat-map-Plot)
  - [Relevant predictors Plot](#Relevant-predictors-Plot)
    
* [Predicting new samples](#Predicting-new-samples)
* [Loading existing model](#Loading-existing-model)


## Creating a model from scratch

* First Clik on *Create new model* button.

![1](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/4b1ddf7f-9ef1-4767-b15c-5c67d6690e35)

* Select a database. The database is the file containing the counting table, which may be a 16-second count of bacteria or other counting data.
  
![2](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/65f63dfe-9429-4ede-8c89-a57ae1eb4e12)

* Browse to where the data is.

![3](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/56e90ea8-c712-4311-88b7-b27a278d27ef)

* Select the database. It can be in biom, qza, csv or tsv formats.

![4](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/54b05cd1-3589-442d-9773-895e3ec76b24)

* You might need to choose the column that corresponds to the sample's id if the data are in csv or tsv file format.
* Simply choose the column from the dropdown menu to accomplish this.

![5](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/6cf42117-4b36-4c4b-b8c7-118399e67e1d)

* Select the Metadata. Is the table with the variable you want to associate with the count table.

![6](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/31075e46-5855-4f4c-88dd-ce2d180076e8)

* Browse to where the metadata is.

![7](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/bda8f337-c788-451e-b693-2ce2ed147763)

* The metadata can be in csv or tsv formats only.

![8](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/a49715fa-a93f-47bd-8e3f-7b018ea2e113)

* The column that corresponds to the sample's id must be chosen.
* Additionally, you need to choose the column that holds the variable (target) you want to associate.

![9](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/035c5d44-7c76-42eb-b6ba-1da3ae93deed)

* After making sure the sample's ID and target are correct, press the *next* button.

![10](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/000e7760-137e-4c19-b4dc-23aeea41f117)

* You can select *Change configurations* to alter the default settings.

![11](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/59a9bf85-33dd-4e72-bbe1-300f399e36da)

* Click the "save" button after making any desired changes to the setup. The model configuration will change.

![12](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/9ad3d1c4-0820-4e7c-9130-8cf7857b8bbb)

* Simply select the "Create Model" option to begin.

![11-5](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/840634d0-dcda-445a-8fd0-1e04ddbc2e30)

* Set the results location and model filename appropriately.

![13](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/3d9195ac-c8a6-4831-9182-0bf92a454f25)

* Await its conclusion.

![14](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/548f35fb-dfc7-4312-99b9-5804372ea8f3)

* The CODARFE results as well as ways to make some graphs can be found on the final scream.
* Every file was created using the distinction you selected.

![15](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/dcb04211-8636-4f03-97ff-26146f11437b)

## Ploting the results

### Correlation Plot

* The correlation between the real and the model's learned data is displayed in this graphic.
* To download just click on the *Download* button.
  
![1](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/1872c5a4-5776-4a0b-a316-c31f82dfd747)

* Select your preferred setup and click the "Save" button.

![2](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/76fc46bf-9b3a-414e-afb6-e53cce6160da)

### Hold Out Validation Plot

* A box-plot with X repetitions of hold-out validation is produced by the hold-out validation.
* On the *Number of repetions* space, choose the repetition count. It must be a higher-order integer than 0.
* From the *Test size* space, choose the test size. The hold portion of the hold-out will be determined by it.
* Select *Plot* after selecting all of the parameters.

  
![3](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/ea439bf2-867f-4503-a98e-5f905cfa9e71)

* Await its conclusion.

![4](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/ee49af1e-e356-4402-a269-7eaf8f5e6aed)

* To download just click on the *Download* button.

![5](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/538e05f8-f4f0-4cce-a9a9-61f76b149d3b)

* Select your preferred setup and click the "Save" button.

![6](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/9ec29a74-601e-45b5-8810-98bb4a4a98a8)

### Heat map Plot

* The *heat-map* is usually too large to be displayed on the screen, so it can only be saved to a file.
* Select your preferred setup and click the "Save" button.
![7](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/cfb0a4ac-2a94-4c8c-a7e5-0684018af04f)

### Relevant predictors Plot

* The top *X* predictors and their correlation strengths and directions will be shown in the relevant predictors plot.
* Using the *Maximum predictors to show* space, choose how many predictors will be presented.


![8](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/720f3b20-4fa2-47d6-aeba-a6658b570fbb)

* To download just click on the *Download* button.

![9](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/884f6b4b-2135-49cd-839e-09e8aa11947a)

* Select your preferred setup and click the "Save" button.

![10](https://github.com/MuriloCaminotto/CODARFE/assets/92797211/fd9d7355-5464-40dc-a4bb-6c85846e4727)



## Predicting new samples

## Loading existing model


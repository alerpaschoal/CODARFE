# CODARFE for Windows users (Visual interface)

The Windows operating system was used in the development of this version of CODARFE, which features a graphical user interface. For those who have had little to no experience with programming, this version is ideal. For each analysis you can perform using this interface, go to the instruction below.

# Limitations
- This version can generate results slightly different from the others due to some limitations during the code development.
- If your target has negative numbers and low variations, it may cause problems during the prediction step. You can *shift* your target by adding the minimum value +1 to all samples, allowing the prediction.
- The data and metadata **MUST NOT contain index as the first columns**.

---

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


![1](https://github.com/user-attachments/assets/d988bc7d-6431-40e0-8e0b-ff7150359dbe)

* Select a database. The database is the file containing the counting table, which may be a 16s count of bacteria or other counting data.
  
![2](https://github.com/user-attachments/assets/0f6fa053-ff82-48a4-9ca2-84ce9a3db30a)


* Browse to where the data is.

![3](https://github.com/user-attachments/assets/5ebb07ca-5ece-49a9-ab24-dec4c37d9dea)


* Select the database. It can be in biom, qza, csv or tsv formats.

![4](https://github.com/user-attachments/assets/d27528fc-820e-4df7-8fe7-c3442898738c)


* You might need to choose the column that corresponds to the sample's id if the data are in csv or tsv file format.
* Simply choose the column from the dropdown menu to accomplish this.


![5](https://github.com/user-attachments/assets/3b94946d-55df-464b-8eaf-a0e2d9e8a91b)

* Select the Metadata. Is the table with the variable you want to associate with the count table.

![6](https://github.com/user-attachments/assets/546c4e0d-7161-4977-ac8a-f79892da95a3)


* Browse to where the metadata is.

![7](https://github.com/user-attachments/assets/da999a57-5abd-4a6e-93b5-de750bdeaa63)

* The metadata can be in csv or tsv formats only.

![8](https://github.com/user-attachments/assets/03d9245f-207c-4765-9455-019e33f6e7b2)

* The column that corresponds to the sample's id must be chosen.
* Additionally, you need to choose the column that holds the variable (target) you want to associate.

![9](https://github.com/user-attachments/assets/967a5143-20a0-4cae-a1be-1e0aa7d3e931)

* After making sure the sample's ID and target are correct, press the *next* button.

![10](https://github.com/user-attachments/assets/3af924d3-20f8-4dc6-a4d2-4e3b9c83f694)

* You can select *Change configurations* to alter the default settings.

![11](https://github.com/user-attachments/assets/7a0af47e-de15-4b7b-8ff3-cb26bf584c49)

* Click the "save" button after making any desired changes to the setup. The model configuration will change.

![12](https://github.com/user-attachments/assets/f0eca987-b20a-4775-b6a6-57cc21989a07)

* Simply select the "Create Model" option to begin.

![11-2](https://github.com/user-attachments/assets/fa314430-868c-41f9-8ab0-55658c3a805d)
* Set the results location and model filename appropriately.

![13](https://github.com/user-attachments/assets/5daec915-a92a-493c-b50c-c9651a9db0f8)

* Await its conclusion.

![14](https://github.com/user-attachments/assets/d79cec6c-8dd9-4f58-8a53-aab8bec1c1f7)

* The CODARFE results can be found on the final scream.

![15](https://github.com/user-attachments/assets/8a83d91d-d53c-4ebe-8718-cdfed422abc5)

## Ploting the results

### Correlation Plot

* The correlation between the real and the model's learned data is displayed in this graphic.
* To download just click on the *Download* button.
  
![1](https://github.com/user-attachments/assets/065d5ee5-c3ef-4ed7-9c15-ecbc4135d623)

* Select your preferred setup and click the "Save" button.

![2](https://github.com/user-attachments/assets/c5546066-245f-4cee-b9c7-516f0b6ea0f7)

### Hold Out Validation Plot

* A box-plot with X repetitions of hold-out validation is produced by the hold-out validation.
* On the *Number of repetions* space, choose the repetition count. It must be a higher-order integer than 0.
* From the *Test size* space, choose the test size. The hold portion of the hold-out will be determined by it.
* Select *Plot* after selecting all of the parameters.

  
![3](https://github.com/user-attachments/assets/bd8cceec-ef6d-47c3-8225-6b35aed4f615)

* Await its conclusion.

![4](https://github.com/user-attachments/assets/59d7d93c-2a23-41c9-b875-c7a816cbbca0)

* To download just click on the *Download* button.

![5](https://github.com/user-attachments/assets/869b2d65-eae5-448f-8b92-64640e1ff22d)

* Select your preferred setup and click the "Save" button.

![6](https://github.com/user-attachments/assets/2489f6ca-ddf0-4308-be7b-1655ddd7c121)

### Heat map Plot

* The *heat-map* is usually too large to be displayed on the screen, so it can only be saved to a file.
* Select your preferred setup and click the "Save" button.

![7](https://github.com/user-attachments/assets/66b46ff7-60e3-4582-af0d-f66e2cc0a314)

### Relevant predictors Plot

* The top *X* predictors and their correlation strengths and directions will be shown in the relevant predictors plot.
* Using the *Maximum predictors to show* space, choose how many predictors will be presented.


![8](https://github.com/user-attachments/assets/914dc400-89e3-4d0f-8658-372a361d7ee3)

* To download just click on the *Download* button.

![9](https://github.com/user-attachments/assets/eedac0a5-a6f2-4361-abee-c462ee0d191f)

* Select your preferred setup and click the "Save" button.

![10](https://github.com/user-attachments/assets/bbb34d30-5802-4746-a068-b18417180685)

## Predicting new samples

* Select the *Predict new samples* button on the final screen.

![1](https://github.com/user-attachments/assets/6c94fc05-8d2e-44d3-a5d8-4bfdecde5aaa)

* Browse in and select the new samples database.

![2](https://github.com/user-attachments/assets/997d4701-2e30-4ca2-8896-9bdec9e83caa)

* If the data are in csv or tsv file format, you may need to select the column that matches the sample's id.
* To do this, just select the column from the dropdown option.
* The file will then be saved together with the model (*.foda) file after you click the "Save" button.


![3](https://github.com/user-attachments/assets/71141477-84c7-454f-b31a-31fb6449979e)

## Loading existing model

* Select the *Upload model* button in the opening scream.

![1](https://github.com/user-attachments/assets/0beb4204-45a5-4786-99c7-ac62df47b332)

* Navigate to the model file (*.foda).

![2](https://github.com/user-attachments/assets/dd2e8711-4f4a-4731-96da-7246c67ef219)

* The model (*.foda) file you want to open should be selected.
* Select the "Open" button.


![3](https://github.com/user-attachments/assets/ff00c21a-388f-4e0e-ad25-18a17f75185f)

* The model's results will be loaded when the final scream appears.

![4](https://github.com/user-attachments/assets/7bc9cdb9-02d4-4b51-b3e8-65daf6752d97)



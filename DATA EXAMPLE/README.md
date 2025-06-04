# Example data for CODARFE

 These records provide the taxonomy of bacteria derived from infats fezes along with the infant's age.

* PREDICTOR_infatn_age.csv -> is the predictor table; has the taxonomy of the bacteria in columns and their abundances in rows.
* METADATA_infat_age.csv -> is the metadata table; contains the target variable, in this case is the infant age. Other columns (which won't be used during CODARFE's training) might be included.

These tables can be used as examples to run any of the CODARFE versions available on the GitHub repository.

# Sources
The dataset was collected from: https://knightslab.org/MLRepo/docs/yatsunenko_infantage.html

The Original paper (first introduced this dataset):
~~~
@article{yatsunenko2012human,
  title={Human gut microbiome viewed across age and geography},
  author={Yatsunenko, Tanya and Rey, Federico E and Manary, Mark J and Trehan, Indi and Dominguez-Bello, Maria Gloria and Contreras, Monica and Magris, Magda and Hidalgo, Glida and Baldassano, Robert N and Anokhin, Andrey P and others},
  journal={nature},
  volume={486},
  number={7402},
  pages={222--227},
  year={2012},
  publisher={Nature Publishing Group UK London}
}
~~~

Additional publicly accessible datasets that were utilized in CODARFE's publish are accessible here: https://doi.org/10.5281/zenodo.12751711

from biom import load_table
import zipfile
import shutil
import os
from skbio.stats.composition import clr
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import operator
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
import sys
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle as pk

# Plots
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from skbio.stats.ordination import ca
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

class CODARFE():
  """
  This class implements a microbiome analysis tool and prediction of continuous environmental variables associated with the microbiome.
  It utilizes the RFE method combined with robust-to-noise regression, CoDA analysis, and 4 metrics for selecting a subgroup of the microbiome highly associated with the target variable.
  As a result, CODARFE can predict the target variable in new microbiome samples.

  CODARFE requires the following parameters:

    path_to_data: str = None
          The path to the microbiome dataframe (counting table)
    flag_first_col_as_index_data: bool = False
          If True, the first row of the dataframe will be used as index
    path_to_metadata: str = None
          The path to the metadata dataframe with the target variable
    flag_first_col_as_index_metaData: bool = False
          If True, the first row of the dataframe will be used as index
    target: str = None
          The name of the target variable column inside the metadata

  Usage:
      1) Create an instance of CODARFE with your data:

      coda = CODARFE(path_to_data       = <path_to_microbiome_dataframe>,
                     path_to_metadata   = <path_to_metadata_dataframe>,
                     target = <string_target_variable_name>)

      2) Train a model:

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


      3) Save the model:

      coda.save_instance(path_out    = <path_to_folder>,
                         name_append = <name>)


      Alternatively, you can load a pre-trained model

      
      coda = CODARFE()
      coda.load_instance(path2instance = <path_to_file_instance.foda>)
     


      4) View the results:

        4.1) Plot of predicted vs. expected correlation

        coda.Plot_Correlation(path_out    = <path_to_folder>,
                              name_append = <name>)

        4.2) Plot the mean absolute error using a hold-out validation

        coda.Plot_HoldOut_Validation( n_repetitions = 100,
                                      test_size     = 20,
                                      path_out      = <path_to_folder>,
                                      name_append   = <name>)

        4.3) Plot of the relationship of predictors with the target

        coda.Plot_Relevant_Predictors(n_max_features = 100,
                                      path_out       = <path_to_folder>,
                                      name_append    = <name>)

        4.4) Heatmap of selected predictors

        coda.Plot_Heatmap(path_out    = <path_to_folder>,
                          name_append = <name>)

        4.5) Selected predictors

        coda.selected_taxa

      5) Predict the target variable in new samples:

      coda.Predict( path_to_new_data = <path_to_new_data>,
                    applyAbunRel = True,
                    writeResults = True,
                    path_out     = <path_out>
                    name_append  = <name>)

  For more information about the tool, visit the original publication or the GitHub containing more versions of this same tool.

  For questions, suggestions, or bug/error reports, contact via email: murilobarbosa@alunos.utfpr.edu.br"

  """
  class ModelNotCreatedError(Exception):
    def __init__(self, mensagem="No model created! Please create the model using the CreateModel function and try again."):
            self.mensagem = mensagem
            super().__init__(self.mensagem)

  class EmptyDataError(Exception):
    def __init__(self, mensagem="No model created! Please create the model using the CreateModel function and try again."):
            self.mensagem = mensagem
            super().__init__(self.mensagem)
            
  def __init__(self,
               path_to_data: Optional[str] = None,
               path_to_metadata: Optional[str] = None,
               target: Optional[str] = None,
               flag_first_col_as_index_data: bool = False,
               flag_first_col_as_index_metaData: bool = False
               ) -> None:
    """
    Parameters
    ----------
    path_to_data: str = None
          The path to the microbiome dataframe (counting table)
    flag_first_col_as_index_data: bool = False
          If True, the first column of the dataframe will be used as index
    path_to_metadata: str = None
          The path to the metadata dataframe with the target variable
    flag_first_col_as_index_metaData: bool = False
          If True, the first column of the dataframe will be used as index
    target: str = None
          The name of the target variable column inside the metadata
    """
    self.__flag_first_col_as_index_data = flag_first_col_as_index_data
    self.__flag_first_col_as_index_metaData = flag_first_col_as_index_metaData
    self.__path_to_metadata = path_to_metadata
    if path_to_data != None and path_to_metadata != None and target != None:
      print("Loading data... It may take some minutes depending on the size of the data")
      self.data, self.target = self.__read_data(path_to_data,path_to_metadata,target)
      self.__totalPredictorsInDatabase = len(self.data.columns)
      print("CODARFE instance created successfully")
    else:
      self.data = None
      self.target = None
      print("New empty instance created")


    self.__sqrt_transform = None
    self.__transform = None
    self.__min_target_sqrt_transformed = None
    self.__max_target_sqrt_transformed = None
    self.__min_target_transformed = None
    self.__max_target_transformed = None
    self.results = None
    self.score_best_model = None
    self.selected_taxa = None
    self.weights = None
    self.__model = None
    self.__n_max_iter_huber = None
    self.__applyAbunRel = True

    self.__correlation_list = {}

  def __read_data(self,path_to_data,path_to_metadata,target_column_name):
    if not os.path.exists(path_to_data):
      print('The Data file does not exists!')
      raise FileNotFoundError('The Data file does not exists!')
    if not os.path.exists(path_to_metadata):
      raise FileNotFoundError('The Metadata file does not exists!')

    extension = path_to_data.split('.')[-1]
    if extension == 'csv':
        data = pd.read_csv(path_to_data,encoding='latin1')
        if self.__flag_first_col_as_index_data:
          data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'tsv':
        data = pd.read_csv(path_to_data,sep='\t',encoding='latin1')
        if self.__flag_first_col_as_index_data:
          data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'biom':
        table = load_table(path_to_data)
        data = table.to_dataframe()
    elif extension == 'qza':
        output_directory =  '/'.join(path_to_data.split('/')[:-1])+'/QZA_EXTRACT_CODARFE_TEMP/'
        # Openning the .qza file as an zip file
        with zipfile.ZipFile(path_to_data, 'r') as zip_ref:
            # extracting all data to the output directory
            zip_ref.extractall(output_directory)
        # Getting the biom path file
        biompath = output_directory+os.listdir(output_directory)[0]+'/data/'
        biompath += [f for f in os.listdir(biompath) if f[-5:]=='.biom'][0]
        table = load_table(biompath)# Read the biom file
        data = table.to_dataframe() # Tranform it to a pandas dataframe

        shutil.rmtree(output_directory) # remove the pathTree created

    extension = path_to_metadata.split('.')[-1]
    if extension == 'csv':
        metadata = pd.read_csv(path_to_metadata,encoding='latin1')
        if self.__flag_first_col_as_index_metaData:
          metadata.set_index(list(metadata.columns)[0],inplace=True)
    elif extension == 'tsv':
        metadata = pd.read_csv(path_to_metadata,sep='\t',encoding='latin1')
        if self.__flag_first_col_as_index_metaData:
          metadata.set_index(list(metadata.columns)[0],inplace=True)

    totTotal = len(metadata)
    totNotNa = metadata[target_column_name].isna().sum()
    metadata.dropna(subset=[target_column_name],inplace=True)

    #metadata = metadata[target_column_name]
    #X = metadata.join(data)
    data = data.loc[metadata.index]
    y = metadata[target_column_name]

    if len(data) == 0:
      raise ValueError('There is no correspondence between the ids of the predictors and the metadata.\nMake sure the column corresponding to the identifiers is first.')

    print('Total samples with the target variable: ',totTotal-totNotNa,'/',totTotal)

    return data,y

  def save_instance(self,path_out: str,name_append: Optional[str]='CODARFE_MODEL')-> None:
    """
    Parameters
    ----------
    path_out: str
              Path to folder where it will be saved. If no path is provided it will save in the same directory as the metadata with the name of 'CODARFE_MODEL.foda'
    name_append: str = 'CODARFE_MODEL'
              Name to concatenate in the final filename.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
              If de path_out does not exists

    """
    if type(self.data) == type(None):
      print('Nothing to save.')
      return

    if '/' in path_out:
      path2 = '/'.join(path_out.split('/')[:-1])
      if not os.path.exists(path2):
        raise FileNotFoundError("The path out does not exists!")

    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/'
    # add '_' if not
    if name_append != '':
      if name_append[0]!= '_':
        name_append = 'CODARFE_MODEL_'+name_append
    else:
      name_append = 'CODARFE_MODEL'
    filename = path_out+name_append+'.foda'
    
    obj = {'data':self.data,
           'target':self.target,
           'sqrt_transform': self.__sqrt_transform,
           'transform': self.__transform,
           'min_target_sqrt_transformed': self.__min_target_sqrt_transformed,
           'max_target_sqrt_transformed': self.__max_target_sqrt_transformed,
           'min_target_transformed': self.__min_target_transformed,
           'max_target_transformed': self.__max_target_transformed,
           'results': self.results,
           'score_best_model': self.score_best_model,
           'selected_taxa': self.selected_taxa,
           'model': self.__model,
           'n_max_iter_huber': self.__n_max_iter_huber,
           'correlation_list':self.__correlation_list,
           'weights':self.weights
          }

    with open(filename,'wb') as f:
      pk.dump(obj,f)

    print('\n\nInstance saved at ',filename,'\n\n')

  def load_instance(self,path2instance: str) -> None:
    """
    Load the CODARFE instance stored in the <path2instance> file into this object.

    Parameters
    ----------
    path2instance: str
                   Path to ".foda" file

    Returns
    ------
    None

    Raises
    ------
    FileNotFoundError
                   If the path2instance does not exists

    """
    if not os.path.exists(path2instance):
      raise FileNotFoundError(f"The file {path2instance} does not exists")

    with open(path2instance,'rb') as f:
      obj = pk.load(f)

    self.data = obj['data']
    self.target = obj['target']
    self.__sqrt_transform = obj['sqrt_transform']
    self.__transform = obj['transform']
    self.__min_target_sqrt_transformed = obj['min_target_sqrt_transformed']
    self.__max_target_sqrt_transformed = obj['max_target_sqrt_transformed']
    self.__min_target_transformed = obj['min_target_transformed']
    self.__max_target_transformed = obj['max_target_transformed']
    self.results = obj['results']
    self.score_best_model = obj['score_best_model']
    self.selected_taxa = obj['selected_taxa']
    self.__model = obj['model']
    self.__n_max_iter_huber = obj['n_max_iter_huber']
    self.__correlation_list = obj['correlation_list']
    self.weights = obj['weights']

    print('\n\nInstance restored successfully!\n\n')


  def __remove_low_var(self):
    aux = self.data.copy()
    cols = aux.columns
    selector = VarianceThreshold(aux.var(axis=1).mean()/8)#8
    aux = selector.fit(aux)
    not_to_drop=list(cols[selector.get_support()])
    totRemoved = len(self.data.columns) - len(not_to_drop)
    print('\nA total of ',totRemoved,' taxa were removed due to very low variance\n')
    self.data =  self.data[not_to_drop]

  def __to_abun_rel(self,data):
    return data.apply(lambda x: x/x.sum() if x.sum()!=0 else x,axis=1)

  # Shift the target to positive numbers only in case it has a CV <= 0.2 and negative numbers, so it can be used in the Poisson distribution (RF) 
  def __calc_new_redimension(self,target):

    min_value = min(target)
    max_value = max(target)
    if self.__min_target_transformed == None or self.__max_target_transformed == None:
      self.__min_target_transformed = min_value
      self.__max_target_transformed = max_value

    resized_numbers = [x + abs(self.__min_target_transformed)+1 for x in target]
    return resized_numbers

  def __calc_inverse_redimension(self,predictions):

    restored_numbers = [x - abs(self.__min_target_transformed)-1 for x in predictions]

    return restored_numbers

  # Uses sqrt to fllaten the target and shift it to 0-100 range in case the CV > 0.2. Helps the model to learn the distribution and solves negative numbers
  def __calc_new_sqrt_redimension(self,target):
    target = target.apply(lambda x: np.sqrt(abs(x)) * (-1 if x < 0 else 1))

    min_value = min(target)
    max_value = max(target)
    if self.__min_target_sqrt_transformed == None or self.__max_target_sqrt_transformed == None:
      self.__min_target_sqrt_transformed = min_value
      self.__max_target_sqrt_transformed = max_value

    new_min = 0  # new min value
    new_max = 100  # new max value
    # resized_numbers = [x + abs(self.__min_target_sqrt_transformed)+1 for x in target]
    resized_numbers = [(x - min_value) / (max_value - min_value) * (new_max - new_min) + new_min for x in target]
    return resized_numbers

  def __calc_inverse_sqrt_redimension(self,predictions):
    new_min = 0  # new min value
    new_max = 100  # new max value

    min_value = self.__min_target_sqrt_transformed
    max_value = self.__max_target_sqrt_transformed

    restored_numbers = [(x - new_min) / (new_max - new_min) * (max_value - min_value) + min_value for x in predictions]

    # restored_numbers = [x + abs(self.__min_target_sqrt_transformed)+1 for x in predictions]
    numeros_restaurados_sqrt_inverse = [(x**2) * (-1 if x <0 else 1) for x in restored_numbers]
    return numeros_restaurados_sqrt_inverse

  def __to_CLR(self,df): # Transform to CLr
    aux = df.copy()
    cols = aux.columns
    aux+=0.0001 # Pseudo count
    aux = clr(aux)
    aux = pd.DataFrame(data=aux,columns=cols)
    return aux

  def __calc_mse_model_centeredv2(self,pred,y,df_model): #pred é o valor predito  # df_model é o número de variáveis
    mean = np.mean(y)
    ssr = sum([(yy-pp)**2 for yy,pp in zip(y,pred)])
    centered_tss = sum([(aux-mean)**2 for aux in y])
    ess = centered_tss - ssr
    return ess/df_model

  def __calc_mse_resid(self,pred,y,df_resid):
    residuos = [yy-pp for yy,pp in zip(y,pred)]
    return sum([aux**2 for aux in residuos])/df_resid

  def __calc_f_prob_centeredv2(self,pred,y,X):
    df_model = max(1,len(X.iloc[0]) - 1)
    df_resid = max(1,len(X) - df_model -1 )
    mse_model = self.__calc_mse_model_centeredv2(pred,y,df_model)
    mse_resid = self.__calc_mse_resid(pred,y,df_resid)
    fstatistic = mse_model/mse_resid
    return stats.f.sf(fstatistic, df_model, df_resid)

  def __calc_r_squared(self,pred,y,method = 'centered'):
    ssr = sum([(yy-pp)**2 for yy,pp in zip(y,pred)])
    mean = np.mean(y)
    center_tss = np.sum((y - mean)**2)
    uncentered_tss = sum([(aux)**2 for aux in y])
    if method == 'centered':
      return 1 - ssr/center_tss
    else:
      return 1 - ssr/uncentered_tss

  def __calc_rsquared_adj(self,X,r2):
    return 1 - (1-r2) * (X.shape[0]-1)/(X.shape[0]-X.shape[1]-1)

  def __calc_llf(self,pred,y,X):
    nobs2 = len(X) / 2.0
    nobs = float(len(X))
    ssr = sum([(yy-pp)**2 for yy,pp in zip(y,pred)])
    llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2
    return llf


  def __calc_bic(self,pred,y,X):
    llf = self.__calc_llf(pred,y,X)
    nobs = float(len(X))
    k_params = len(X.iloc[0])
    bic = -2*llf + np.log(nobs) * k_params
    return round(bic)

  def __check_coef_of_variation_target_for_transformation(self,allow_transform_high_variation):
    if allow_transform_high_variation and np.std(self.target)/np.mean(self.target)>0.2 :# Caso varie muitas vezes a média (ruido)
      
      self.__sqrt_transform = True 
      self.__transform = False

    else:
      if any(t < 0 for t in self.target):
        self.__transform = True
        self.__sqrt_transform = False
        
      else:
        self.__sqrt_transform = False # Define flag de transformação
        self.__transform = False

  def __super_RFE(self,method,n_cols_2_remove,n_Kfold_CV):
    # Sets the total number of attributes to be removed per round
    n_cols_2_remove = max([int(len(self.data.columns)*n_cols_2_remove),1])

    # Set initial X as descriptors
    if self.__applyAbunRel:
      X_train = self.__to_abun_rel(self.data)
    else:
      X_train = self.data

    # Defines the target, with the proper transformation
    if self.__sqrt_transform:
      y_train = np.array(self.__calc_new_sqrt_redimension(self.target))
    elif self.__transform:
      y_train = np.array(self.__calc_new_redimension(self.target))
    else:
      y_train = self.target

    # Initialize results table
    resultTable = pd.DataFrame(columns=['Atributos','R² adj','F-statistic','BIC','MSE-CV'])
    percentagedisplay = round(len(self.data.columns) - (len(X_train.columns)/len(self.data.columns)))
    while len(X_train.columns) - n_cols_2_remove > n_cols_2_remove or len(X_train.columns) > 1:
      X = self.__to_CLR(X_train)

      method.fit(X,y_train)

      pred = method.predict(X)

      Fprob = self.__calc_f_prob_centeredv2(pred,y_train,X)

      r2 = self.__calc_r_squared(pred,y_train)
      r2adj = self.__calc_rsquared_adj(X,r2)

      BIC = self.__calc_bic(pred,y_train,X)

      msecv_results = []
      #Add cross-val validation for RMSE 
      kf = KFold(n_splits=n_Kfold_CV,shuffle=True,random_state=42)

      for train, test in kf.split(X):

        if isinstance(y_train,pd.core.series.Series):
          model = method.fit(X.iloc[train],y_train.iloc[train])

          predcv = model.predict(X.iloc[test])

          msecv_results.append(np.mean([(t-p)**2 for t,p in zip(y_train.iloc[test],predcv)])**(1/2))
        else:
          model = method.fit(X.iloc[train],y_train[train])

          predcv = model.predict(X.iloc[test])

          msecv_results.append(np.mean([(t-p)**2 for t,p in zip(y_train[test],predcv)])**(1/2))

      msecv = np.mean(msecv_results)

      # if there is a p_value
      if not math.isnan(Fprob) and r2adj < 1.0:
        # create a new line separeted by ";" with: R² adj, F-statistic and BIC
        newline = pd.DataFrame.from_records([{'Atributos':'@'.join(list(X.columns)),'R² adj':float(r2adj),'F-statistic':float(Fprob),'BIC':float(BIC),'MSE-CV':float(msecv)}])
        resultTable = pd.concat([resultTable,newline])

      # Create table of attributes
      atr = pd.DataFrame(data = [[xx,w] for xx,w in zip(X.columns,method.coef_)],columns = ['Atributos','coef'])

      atr = atr.astype({'coef':float})

      #Remove columns with coef 0 to speed up RFE
      atr = atr[atr.coef != 0]# 
      atr.coef = atr.coef.abs()# Transform it to abs to remove only near zero values (keep negative high correlation)


      atr = atr.sort_values(by=['coef'],ascending=False)

      #Create list of selected attributes
      atributos = list(atr.Atributos)

      
      atributos = [x.strip() for x in atributos]

      # Remove n_cols_2_remove lowest coefs
      atributos = atributos[:-n_cols_2_remove]

      # Remove not selected
      if self.__applyAbunRel:
        X_train = self.__to_abun_rel(self.data[atributos])
      else:
        X_train = X_train[atributos]

      # Calculates the % for display ... not continued in this version
      percentagedisplay = round(len(self.data.columns) - (len(X_train.columns)/len(self.data.columns)))
      #print(percentagedisplay,'% done...\n')
    
    resultTable.dropna(inplace=True)
    #print('100% done!\n')
    
    return resultTable

  def __score_and_selection(self,resultTable,weightR2,weightProbF,weightBIC,weightRMSE):
    # copy original
    df_aux = resultTable.copy()
    df_aux.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_aux.dropna(inplace=True)

    # Normalize R² adj
    df_aux['R² adj'] = MinMaxScaler().fit_transform(np.array(df_aux['R² adj']).reshape(-1,1))

    # Apply -log10(f) and normalize F-statistic
    df_aux['F-statistic'] = [-math.log10(x) if x !=0 else sys.float_info.max for x in df_aux['F-statistic']]
    df_aux['F-statistic'] = MinMaxScaler().fit_transform(np.array(df_aux['F-statistic']).reshape(-1,1))

    # Normalize and inverts BIC (lower the better)
    df_aux['BIC'] = MinMaxScaler().fit_transform(np.array(df_aux['BIC']).reshape(-1,1))
    df_aux['BIC'] = [np.clip(1-x,0,1) for x in df_aux['BIC']]

    # Normalize and inverts 'MSE-CV'
    df_aux['MSE-CV'] = MinMaxScaler().fit_transform(np.array(df_aux['MSE-CV']).reshape(-1,1))
    df_aux['MSE-CV'] = [np.clip(1-x,0,1) for x in df_aux['MSE-CV']]

    # Create Score column
    df_aux['Score'] = [(r*weightR2)+(f*weightProbF)+(b*weightBIC)+(m*weightRMSE) for r,f,b,m in zip(df_aux['R² adj'],df_aux['F-statistic'],df_aux['BIC'],df_aux['MSE-CV'])]

    # Finds the hights score index
    indexSelected = list(df_aux.Score).index(max(list(df_aux.Score)))

    # Select attributes
    selected = df_aux.iloc[indexSelected].Atributos.split('@')

    self.selected_taxa = selected # save it

    retEstatisticas = list(resultTable.iloc[indexSelected][['R² adj','F-statistic','BIC','MSE-CV']])

    self.results = {'R² adj':retEstatisticas[0],
                    'F-statistic':retEstatisticas[1],
                    'BIC':retEstatisticas[2],
                    'MSE-CV':retEstatisticas[3]} 

    retScore = df_aux.iloc[indexSelected].Score

    self.score_best_model = retScore # 

  def __write_results(self,path_out,name_append):
    # adiciona '/' caso n tneha
    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if type(self.__path_to_metadata) != type(None):
        if self.__path_to_metadata.split('/')[:-1] != []:
          path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/'

    # add '_' if not
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append

    path2write = path_out+'CODARFE_RESULTS' +name_append+'.txt'
    print('Writing results at ',path2write)

    with open(path2write,'w') as f:
      f.write('Results: \n\n')
      f.write('R² adj -> '+     str(self.results['R² adj'])+'\n')
      f.write('F-statistic -> '+str(self.results['F-statistic'])+'\n')
      f.write('BIC -> '+        str(self.results['BIC'])+'\n')
      f.write('MSE-CV -> '+     str(self.results['MSE-CV'])+'\n')
      f.write('Total of taxa selected -> '+str(len(self.selected_taxa))+'. This value corresponds to '+str((len(self.selected_taxa)/len(self.data.columns))*100)+'% of the total observed\n\n')
      f.write('Selected taxa: \n\n')
      f.write(','.join(self.selected_taxa))

  def __define_model(self,allow_transform_high_variation):

      self.__model = RandomForestRegressor(n_estimators = 160, criterion = 'poisson',random_state=42)
      X = self.data[self.selected_taxa]
      if self.__applyAbunRel:
        X = self.__to_abun_rel(X)
      X = self.__to_CLR(X)

      if self.__sqrt_transform:# Caso varie muitas vezes a média (ruido)
      
        targetLogTransformed = self.__calc_new_sqrt_redimension(self.target) # Aplica transformação no alvo
        self.__model.fit(X,targetLogTransformed) # Treina com o alvo transformado
        
      elif self.__transform:

        targetTranformed = self.__calc_new_redimension(self.target)
        self.__model.fit(X,targetTranformed)
        print(f"The data was shifted {abs(self.__min_target_transformed)} + 1 units due to negative values not supported by poisson distribution.")

      else:
        self.__model.fit(X,self.target) # Treina um segundo modelo com o alvo como é
  

  def __check_boolean(self,value, name):
      if not isinstance(value, bool):
          raise ValueError(f"{name} must be a boolean.")

  def __check_string(self,value, name):
      if not isinstance(value, str):
          raise ValueError(f"{name} must be a string.")

  def __check_integer_range(self,value, name, min_value, max_value):
      if value < min_value or value > max_value:
          raise ValueError(f"{name} must be between {min_value} and {max_value}.")

  def __check_integer(self,value, name):
      if not isinstance(value, int):
          raise ValueError(f"{name} must be an integer.")

  def __check_non_negative_float(self,value, name):
      if not isinstance(value, float):
          raise ValueError(f"{name} must be a float.")
      elif value < 0:
          raise ValueError(f"{name} must be greater than or equal to 0.")

  def __check_model_params(self,
                         write_results,
                         path_out,
                         name_append,
                         rLowVar,
                         applyAbunRel,
                         allow_transform_high_variation,
                         percentage_cols_2_remove,
                         n_Kfold_CV,
                         weightR2,
                         weightProbF,
                         weightBIC,
                         weightRMSE,
                         n_max_iter_huber):
      self.__check_boolean(write_results, "write_results")
      self.__check_boolean(rLowVar, "rLowVar")
      self.__check_boolean(applyAbunRel, "applyAbunRel")
      self.__check_boolean(allow_transform_high_variation, "allow_transform_high_variation")
      self.__check_string(path_out, "path_out")
      if write_results and path_out != '' and not os.path.exists(path_out):
          raise FileNotFoundError("The path out does not exist!")
      self.__check_string(name_append, "name_append")
      self.__check_integer(percentage_cols_2_remove, "percentage_cols_2_remove")
      self.__check_integer_range(percentage_cols_2_remove, "percentage_cols_2_remove", 1, 99)
      self.__check_integer(n_Kfold_CV, "n_Kfold_CV")
      self.__check_integer_range(n_Kfold_CV, "n_Kfold_CV", 2, 100)
      self.__check_integer(n_max_iter_huber, "n_max_iter_huber")
      self.__check_integer_range(n_max_iter_huber, "n_max_iter_huber", 2, 1000)
      self.__check_non_negative_float(weightR2, "weightR2")
      self.__check_non_negative_float(weightProbF, "weightProbF")
      self.__check_non_negative_float(weightBIC, "weightBIC")
      self.__check_non_negative_float(weightRMSE, "weightRMSE")

  def fit(self,
                  write_results: bool =True,
                  path_out: str ='',
                  name_append: str ='',
                  rLowVar: bool =True,
                  applyAbunRel: bool = True,
                  allow_transform_high_variation: bool = True,
                  percentage_cols_2_remove: int =1,
                  n_Kfold_CV: int=10,
                  weightR2: int =1.0,
                  weightProbF: float=0.5,
                  weightBIC: float=1.0,
                  weightRMSE: float=1.5,
                  n_max_iter_huber: int=100):


    """
    Parameters
    ----------
    write_results:  bool = False
                    Defines if the results will be written. The results include the selected predictors and the metrics for its selection.
    path_out: str = ""
                    Where to write the results
    name_append: str = ""
                    The name to append in the end of the file with the results
    rLowVar: bool = True
                    Flag to define if it is necessary to apply the removal of predictors with low variance. Set as False if less than 300 predictors.
    applyAbunRel: bool = True
                    Flag to define if it is necessary to apply the relative abundance transformation. Set as False if the data is already transformed
    allow_transform_high_variation: bool = True
                    Flag to allow the target transformation in case it has a high variance.
    percentage_cols_2_remove: int = 1
                    Percentage of the total predictors removed in each iteraction of the RFE. HIGH IMPACT in the final result and computational time.
    n_Kfold_CV: int = 10
                    Number of folds in the Cross-validation step for the RMSE calculation. HIGH IMPACT in the final result and computational time.
    weightR2: float = 1.0
                    Weight of the R² metric in the model’s final score
    weightProbF: float = 0.5
                    Weight of the Probability of the F-test metric in the model’s final score
    weightBIC: float = 1.0
                    Weight of the BIC metric in the model’s final score
    weightRMSE: float = 1.5
                    Weight of the RMSE metric in the model’s final score
    n_max_iter_huber: int = 100
                    Maximum number of iterations of the huber regression. HIGH IMPACT in the final result and computational time.

    Returns
    -------
    None

    Raises
    ------
    ValueError
                    If any of the parameters is not the correct type or is outside the range

    """

    if type(self.data) == type(None):
      print('No data was provided!\nPlease make sure to provide complete information or use the load_instance(<path_2_instance>) function to load an already created CODARFE model')
      return None
    print('\n\nChecking model parameters...',end="")
    self.__check_model_params(write_results,path_out,name_append,rLowVar,applyAbunRel,allow_transform_high_variation,percentage_cols_2_remove,n_Kfold_CV,weightR2,weightProbF,weightBIC,weightRMSE,n_max_iter_huber)
    print('OK')

    n_cols_2_remove = percentage_cols_2_remove/100
    self.__n_max_iter_huber = n_max_iter_huber # Defines the number of iterations used by hubber regressor

    if rLowVar:
      #Removes low variation
      self.__remove_low_var()

    if applyAbunRel:
      #transform to abunrel
      self.__applyAbunRel = True

    method = HuberRegressor(epsilon = 2.0,alpha = 0.0003, max_iter = n_max_iter_huber)
    # Define flags related to target transformation
    self.__check_coef_of_variation_target_for_transformation(allow_transform_high_variation)

    # Iteratively remove attributes while creating multiple models
    resultTable = self.__super_RFE(method,n_cols_2_remove,n_Kfold_CV)

    if len(resultTable)>0:
      # Calcula pontuação e seleciona o melhor modelo
      self.__score_and_selection(resultTable,weightR2,weightProbF,weightBIC,weightRMSE)

      self.__define_model(allow_transform_high_variation)

      print('\nModel created!\n\n')
      print('Results: \n\n')
      print('R² adj -> ',     self.results['R² adj'])
      print('F-statistic -> ',self.results['F-statistic'])
      print('BIC -> ',        self.results['BIC'])
      print('MSE-CV -> ',     self.results['MSE-CV'])
      print('Total of taxa selected -> ',len(self.selected_taxa),'. This value corresponds to ',(len(self.selected_taxa)/self.__totalPredictorsInDatabase)*100,'% of the total.\n')

      # Estou adicionando isso muitos meses apos ter feito o codigo... pra garantir que n vou quebrar nada, eu so re-treino o modelo e salvo os pesos
      method = HuberRegressor(epsilon = 2.0,alpha = 0.0003, max_iter = self.__n_max_iter_huber)
      X = self.data[self.selected_taxa]
      if self.__applyAbunRel:
        X = self.__to_abun_rel(X)
      X = self.__to_CLR(X)
      y = self.target
      resp = method.fit(X,y)
      self.weights = pd.DataFrame(data={'Predictors':resp.feature_names_in_,'weights':resp.coef_})
      # ---------------------------------------------------------------------------------------------------------------------------------------------
      
      if write_results:
        self.__write_results(path_out,name_append)

    else:
      print('The model was not able to generalize your Data.')

  def __pairwise_correlation(self,A, B):
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm /  (np.sqrt(
        np.sum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.sum(bm**2, axis=0, keepdims=True)))

  def __create_correlation_imputer(self):
    threshold = 0.7 # Considered as strong correlation
    aux = self.__to_CLR(self.data) # Apply CLR

    for selected in self.selected_taxa: # For each taxa
      self.__correlation_list[selected] = [] # create a instance of this taxa
      for taxa in aux.columns: # verify correlation to every single other taxa 
        if taxa != selected: # 
          corr = self.__pairwise_correlation(np.array(aux[selected]),np.array(aux[taxa]))# calculates the correlation
          if corr >= threshold: # adds it if high correlated
            self.__correlation_list[selected].append({'taxa':taxa,'corr':corr}) # add the new taxa
      self.__correlation_list[selected].sort(reverse=True,key = lambda x: x['corr']) # Sort by correlation

  def __read_new_data(self,path_to_data,flag_first_col_as_index):
    extension = path_to_data.split('.')[-1]
    if extension == 'csv':
        data = pd.read_csv(path_to_data,encoding='latin1')
        if flag_first_col_as_index:
          data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'tsv':
        data = pd.read_csv(path_to_data,sep='\t',encoding='latin1')
        if flag_first_col_as_index:
          data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'biom':
        table = load_table(path_to_data)
        data = table.to_dataframe()
    elif extension == 'qza':
        output_directory =  '/'.join(path_to_data.split('/')[:-1])+'/QZA_EXTRACT_CODARFE_TEMP/'
        # Openning the .qza file as an zip file
        with zipfile.ZipFile(path_to_data, 'r') as zip_ref:
            # extracting all data to the output directory
            zip_ref.extractall(output_directory)
        # Getting the biom path file
        biompath = output_directory+os.listdir(output_directory)[0]+'/data/'
        biompath += [f for f in os.listdir(biompath) if f[-5:]=='.biom'][0]
        table = load_table(biompath)# Read the biom file
        data = table.to_dataframe() # Tranform it to a pandas dataframe

        shutil.rmtree(output_directory) # remove the pathTree created
    return data

  def predict(self,
              path_to_new_data: str,
              applyAbunRel: bool = True,
              writeResults: bool = True,
              flag_first_col_as_index = True,
              path_out: str = '',
              name_append: str = ''
              ) -> Optional[Tuple[pd.DataFrame,int]]:
    """
    Parameters
    ---------
    new : str
          The path to the dataframe with new samples for predicting the target variable
    applyAbunRel: bool = True
          Flag to apply relative abundance transformation
    writeResults: bool = False
          Flag to write the results
    flag_first_col_as_index: bool = True
          Flag indicating if the index appears as the first column in the new data file
    path_out: str = ""
          Filename of the output. If no filename is provided it will be saved in the same directory as the metadata with the name of 'Prediction.csv'
    name_append: str = ""
          Name to concatenate in the final filename. (Use it to differentiate predictions from the same model)

    Returns
    -------
      Tuple with a Pandas Dataframe and a integer
        Pandas Dataframe: Two columns: index and predicts
        Integer : The number of predictors that were missing from the new samples (higher the number, higher the error chance; refer to the original paper)

      If the new data contains fewer than 25% of the total predictors, no prediction is made and None is returned.

    Raises
    -------
      ModelNotCreatedError:
        If the model was not created yet
      FileNotFoundError:
        If writeResults is True but there is no path_out or path_out does not exists
    """
    if self.__model == None:
      raise self.ModelNotCreatedError()

    if writeResults and ((path_out != '' and not os.path.exists(path_out)) or path_out == ''):
      raise FileNotFoundError('\nThe path out does not exists or is empty.')

    new = self.__read_new_data(path_to_new_data,flag_first_col_as_index)
    newindex = new.index
    if self.__correlation_list == {}:
      print('\n\nCreating correlation list for imputation method. It may take a few minutes depending on the size of the original dataset, but it will be created only once.\n\n')
      self.__create_correlation_imputer()
      print('Correlation list created!\n\n')

    data2predict = pd.DataFrame() # Creates a dataframe to place only the selected data
    totalNotFound = 0
    for selected in self.selected_taxa: # For each selected taxa
      if selected in new.columns: # if exists in the new samples
        data2predict[selected] = new[selected] # add the new value
      else: # Senão
        found = False # Flag that defines if a substitute was found
        for correlated_2_selected in self.__correlation_list[selected]: #For each taxa
          if correlated_2_selected['taxa'] in new.columns: # if a substitute was found
            replacing = new[correlated_2_selected['taxa']] # Coloca ele no lugar do que n existe
            data2predict[selected] = replacing
            found = True # Seta flag
            print(f"Taxa {selected} was replaced by {replacing}")
            break
        if not found:
          data2predict[selected] = 0 # if could not find one, retorn zero
          print('Warning! Taxa ',selected,' was not found and have no correlations! It may affect the model accuracy')
          totalNotFound+=1

    if totalNotFound >= len(self.selected_taxa)*0.75:
      print('The new samples has less than 25% of selected taxa. The model will not be able to predict it.')
      return None,totalNotFound

    
    data2predict = data2predict.fillna(0)

    if applyAbunRel:
      data2predict = self.__to_abun_rel(data2predict) 


    data2predict = self.__to_CLR(data2predict) 

    resp = self.__model.predict(data2predict)

    if self.__sqrt_transform: # If the model was trained using sqrt 
      resp = self.__calc_inverse_sqrt_redimension(resp)
    if self.__transform:
      resp = self.__calc_inverse_redimension(resp) # undo transformation

    if writeResults:

      if path_out != '':
        if path_out[-1]!= '/':
          path_out+='/Prediction'
      else:
        path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/Prediction'
      if name_append != '':
        name_append = '_'+name_append
      filename = path_out+name_append+'.csv'
      pd.DataFrame(data = resp,columns = ['Prediction'],index=newindex).to_csv(filename)

    return resp,totalNotFound

  def plot_correlation(self,saveImg: bool=False,path_out: str='',name_append: str='') -> None:
    """
    Parameters
    ---------
    saveImg:  bool = False
              Flag that defines if the img will be saved
    path_out: str = ""
              The path to the folder where the img will be saved
    name_append: str = ""
              The name to append in the end of the img name (Correlation_<name_append>)

    Returns
    ------
    None

    Raises
    ------
    ModelNotCreatedError:
              if the CODARFE.CreateModel was not run yet
    FileNotFoundError:
              If the path_out does not exists
    """
    if self.__model == None:
      raise self.ModelNotCreatedError()

    if path_out != '' and not os.path.exists(path_out):
      raise FileNotFoundError("\nThe path out does not exists.\nPlease try again with the correct path or let it blank to write in the same path as the metadata")

    # Build a rectangle in axes coords
    left, width = .15, .75
    bottom, height = .15, .75
    right = left + width
    top = bottom + height
    y = self.target
    X = self.data[self.selected_taxa]

    if self.__applyAbunRel:
      X = self.__to_abun_rel(X)

    X = self.__to_CLR(X)
    pred = self.__model.predict(X)

    if self.__sqrt_transform: # Caso tenha aprendido com valores transformados
      pred = self.__calc_inverse_sqrt_redimension(pred) # Destransforma-os
    if self.__transform:
      pred = self.__calc_inverse_redimension(pred) # Destransforma-os

    plt.figure()
    plt.clf()
    ax = plt.gca()

    corr, what = pearsonr(y, pred)

    #Plota os pontos previsto por esperado
    plt.plot(pred, y, 'o')

    # calcula o slop e intercept para uma regressão linear (plot da linha)
    m, b = np.polyfit(pred, y, 1)

    #Adiciona a linha no plot
    plt.plot(pred, m*pred+b)
    shiftX = 0.2 * max(pred)
    shiftY = 0.1 * max(y)

    ax.text(left, top, 'R = '+str(round(corr,2))+', p < '+str(round(what,2)),
          horizontalalignment='center',
          verticalalignment='center',
          transform=ax.transAxes)

    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if type(self.__path_to_metadata) != type(None):
        if self.__path_to_metadata.split('/')[:-1] != []:
          path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/'
    # add '_' if not
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append

    filename = path_out+'Correlation'+name_append+'.png'

    print('\nSaving the image at ',filename)
    plt.savefig(filename, dpi=600, bbox_inches='tight')

  def __checkHoldOutParams(self,n_repetitions,test_size,path_out,name_append):
    self.__check_integer(n_repetitions,"n_repetitions")
    self.__check_integer_range(n_repetitions,"n_repetitions",2,1000)
    self.__check_integer(test_size,"test_size")
    self.__check_integer_range(test_size,"test_size",1,99)
    if path_out != '' and not os.path.exists(path_out):
      raise FileNotFoundError("\nThe path out does not exists.\nPlease try again with the correct path or let it blank to write in the same path as the metadata")

  def plot_holdOut_validation(self,
                              n_repetitions: int = 100,
                              test_size: int = 20,
                              saveImg: str = False,
                              path_out: str = '',
                              name_append: str = '') -> None:

    """
    Parameters
    ---------
    n_repetitions:  int = 100
                    Defines the number of repetitions (dots in the plot)
    test_size: int = 20
                    Defines the size of the hold-out samples
    saveImg:  bool = False
                    Flag that defines if the img will be saved
    path_out: str = ""
                    The path to the folder where the img will be saved
    name_append: str = ""
                    The name to append in the end of the img name (HoldOut_Validation_<name_append>)

    Returns
    -------
    None

    Raises
    ------
    ValueError
              If any of the parameters is not the correct type or is outside the range
    ModelNotCreatedError
              if the CODARFE.CreateModel was not run yet
    FileNotFoundError:
              If the path_out does not exists
    """
    if self.__model == None:
      raise self.ModelNotCreatedError()

    self.__checkHoldOutParams(n_repetitions,test_size,path_out,name_append)
    
    method = RandomForestRegressor(n_estimators = 160, criterion = 'poisson',random_state=42)
    X = self.data[self.selected_taxa]
    
    if self.__applyAbunRel:
      X = self.__to_abun_rel(X)

    X = self.__to_CLR(X)
    y = self.target
    maes = []
    for i in range(n_repetitions):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) # divide em treino e teste

      if self.__sqrt_transform: # Caso tenha aprendido originalmente com valores transformados
        method.fit(X_train,self.__calc_new_sqrt_redimension(y_train)) # Re-treina com os valores transformados
        y_pred = method.predict(X_test) # Realiza a predição
        y_pred = self.__calc_inverse_sqrt_redimension(y_pred) # Destransforma-os
      elif self.__transform:
        method.fit(X_train,self.__calc_new_redimension(y_train))
        y_pred = method.predict(X_test)
        y_pred = self.__calc_inverse_redimension(y_pred)
      else:
        method.fit(X_train,y_train)
        y_pred = method.predict(X_test)

      # Calculo do MAE
      tt = 0
      for ii in range(len(y_pred)):
        tt+=abs(y_test.iloc[ii]-y_pred[ii])
      maes.append(tt/len(y_pred))

    sns.set_theme(style="ticks")

    # Cria a figura zerada
    plt.figure()
    f, ax = plt.subplots(figsize=(7, 6))

    # Plota o boxplot
    sns.boxplot(x=[1]*len(maes),
                y=maes,
                whis=[0, 100],
                width=.6,
                palette="vlag")

    # Adiciona os pontos sobre o boxplot
    sns.stripplot(x=[1]*len(maes),
                  y=maes,
                  size=4,
                  color=".3",
                  linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)

    trainSize = int((1-test_size) *100)
    testSize = int(test_size*100)
    ax.set_title('Hold-out Validation ('+str(trainSize)+'-'+str(testSize)+') '+str(n_repetitions)+' repetitions',fontweight='bold')
    ax.set_ylabel('Mean Absolute Error')
    #ax.set_xlabel(target+' MAE')

    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if type(self.__path_to_metadata) != type(None):
        if self.__path_to_metadata.split('/')[:-1] != []:
          path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/'
    # add '_' if not
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append
    
    filename = path_out+'HoldOut_Validation'+name_append+'.png'

    print('\nSaving the image at ',filename)
    plt.savefig(filename, dpi=600, bbox_inches='tight')


  def plot_relevant_predictors(self,
                               n_max_features: int = 100,
                               saveImg: bool = False,
                               path_out: str = '',
                               name_append: str = '') -> None:
    """
    Parameters
    ---------
    n_max_features: int = 100
                    Defines the maximum number of features/predictors to be displayed (bars in the plot)
    saveImg:  bool = False
                    Flag that defines if the img will be saved
    path_out: str = ""
                    The path to the folder where the img will be saved
    name_append: str = ""
                    The name to append in the end of the img name (HoldOut_Validation_<name_append>)

    Returns
    -------
    None

    Raises
    ------
    ValueError
              If any of the parameters is not the correct type or is outside the range
    ModelNotCreatedError
              if the CODARFE.CreateModel was not run yet
    FileNotFoundError:
              If the path_out does not exists
    """

    if self.__model == None:
      raise self.ModelNotCreatedError()
    self.__check_integer(n_max_features,"n_max_features")
    self.__check_integer_range(n_max_features,"n_max_features",2,1000)

    if path_out != '' and not os.path.exists(path_out):
      raise FileNotFoundError("\nThe path out does not exists.\nPlease try again with the correct path or let it blank to write in the same path as the metadata")

    method = HuberRegressor(epsilon = 2.0,alpha = 0.0003, max_iter = self.__n_max_iter_huber)
    X = self.data[self.selected_taxa]
    if self.__applyAbunRel:
      X = self.__to_abun_rel(X)
    X = self.__to_CLR(X)
    # y = self.target
    if self.__sqrt_transform:
      y = np.array(self.__calc_new_sqrt_redimension(self.target))
    elif self.__transform:
      y = np.array(self.__calc_new_redimension(self.target))
    else:
      y = self.target
    resp = method.fit(X,y)

    dfaux = pd.DataFrame(data={'features':resp.feature_names_in_,'coefs':resp.coef_})
    dfaux.sort_values(by='coefs',ascending=False,inplace=True,ignore_index=True)

    if len(dfaux) > n_max_features:
      half = int(n_max_features/2)
      totpos = len(dfaux.coefs[dfaux.coefs>0])
      totneg = len(dfaux.coefs[dfaux.coefs<0])

      if totpos < half:
        totneg = half+half-totpos
      elif totneg < half:
        totpos = half+half-totneg
      else:
        totpos = half
        totneg = half

      dfaux = dfaux[dfaux.index.isin([i for i in range(0,totpos)] + [i for i in range(len(dfaux)-totneg,len(dfaux))])]
    plt.figure()
    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))#figsize=(6, 15)

    colors = ['b' if x > 0 else 'r' for x in dfaux.coefs]
    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="coefs",
                y="features",
                data=dfaux,
                palette=colors,
                )

    ax.set_title('Strength of relevant predictors',fontweight='bold')
    ax.set(ylabel="Predictor name",
          xlabel="Coefficient weight")
    sns.despine(left=True, bottom=True)

    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if type(self.__path_to_metadata) != type(None):
        if self.__path_to_metadata.split('/')[:-1] != []:
          path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/'
    # add '_' if not
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append
    
    filename = path_out+'Relevant_Predictors'+name_append+'.png'


    print('\nSaving the image at ',filename)
    plt.savefig(filename, dpi=600, bbox_inches='tight')


  # HEAT MAP (ps: eu n lembro de porra nenhuma de como eu criei isso... melhor n tentar otimizar nada)
  def __neatMapLinkage(self,selected_features):
    selected_features+=0.001
    w = ca(selected_features)
    pc1 = w.features['CA1']
    pc2 = w.features['CA2']

    xc = np.mean(pc1)
    yc = np.mean(pc2)
    theta = []
    for i in range(len(pc1)):
      theta.append(math.atan2(pc2[i] - yc, pc1[i] - xc ))
    order = [index for index, element in sorted(enumerate(theta), key=operator.itemgetter(1))]
    names = [selected_features.columns[i] for i in order]
    return names

  def __heatmap(self,data, row_labels, col_labels, ax=None,cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = ax.figure.colorbar(im, ax=ax,cax=cax, **cbar_kw)#im
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticklabels(labels=col_labels)
    ax.set_yticklabels(labels=row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
              rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

  def plot_heatmap(self,
                   saveImg: bool=False,
                   path_out: str='',
                   name_append: str=''
                   ) -> None:
    """
    Parameters
    ---------
    saveImg:  bool = False
              Flag that defines if the img will be saved
    path_out: str = ""
              The path to the folder where the img will be saved
    name_append: str = ""
              The name to append in the end of the img name (HeatMap_<name_append>)

    Returns
    -------
    None

    Raises
    ------
    ModelNotCreatedError
              if the CODARFE.CreateModel was not run yet
    FileNotFoundError:
              If the path_out does not exists
    """

    if self.__model == None:
      raise self.ModelNotCreatedError()

    if path_out != '' and not os.path.exists(path_out):
      raise FileNotFoundError("\nThe path out does not exists.\nPlease try again with the correct path or let it blank to write in the same path as the metadata")

    # Pega o dataframe original porem apenas o que foi selecioando
    selected_features = self.data[self.selected_taxa]
    if self.__applyAbunRel:
      selected_features = self.__to_abun_rel(selected_features)
    # Clusterizando bacterias
    y = self.target

    ###### Aqui clusteriza por CA ############
    leaf_names = self.__neatMapLinkage(selected_features)
    ##########################################
    clustered_df = pd.DataFrame()

    for name in leaf_names:
      clustered_df[name] = selected_features[name]
    clustered_df['Target'] = y
    selected_features = clustered_df

    # Ordenando bacterias por variável alvo
    selected_features_t = selected_features.T
    sorted_t = selected_features_t.sort_values(by='Target',axis=1,ascending=False)
    y = list(sorted_t.iloc[-1])

    # Separando os dados para o plot
    bac_counts = sorted_t.drop('Target',axis=0).replace(0,0.5).values

    bacs = list(sorted_t.drop('Target',axis=0).index[:])

    # Aplica o CLR
    bac_clr = clr(bac_counts+0.001)
    vmin = min(bac_clr.flatten())
    vmax = max(bac_clr.flatten())
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    Largura = int(len(y)*0.2)
    Altura  = int(len(bac_counts)*0.2)
    if Largura < 15:
      Largura = 15
    if Altura < 20:
      Altura = 20
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(Largura,Altura))

    im, cbar = self.__heatmap(bac_clr, bacs, y, ax=ax, cmap="RdYlBu",norm = norm, cbarlabel="Center-Log-Ratio")

    fig.tight_layout()


    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if type(self.__path_to_metadata) != type(None):
        if self.__path_to_metadata.split('/')[:-1] != []:
          path_out = '/'.join(self.__path_to_metadata.split('/')[:-1])+'/'
    # add '_' if not
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append
    filename = path_out+'HeatMap'+name_append+'.png'


    print('\nSaving the image at ',filename)
    plt.savefig(filename, dpi=600, bbox_inches='tight')

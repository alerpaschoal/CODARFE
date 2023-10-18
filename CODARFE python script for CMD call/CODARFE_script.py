class CODARFE():
  def __init__(self,
               #X: pd.core.frame.DataFrame=None,
               #y: pd.core.series.Series=None
               path2Data = None,
               path2MetaData = None,
               metaData_Target=None
               ):
    
    if path2Data != None and path2MetaData != None and metaData_Target != None:
      self.metaData_Target = metaData_Target
      print("Loading data... It may take some minutes depending on the size of the data")
      self.data, self.target , self.metadata = self.__Read_Data(path2Data,path2MetaData,metaData_Target)
      self.__totalPredictorsInDatabase = len(self.data.columns)
      self.__path2MetaData = path2MetaData
    else:
      self.data = None
      self.target = None
      self.__path2MetaData = None
      # print('No complete data provided. Please use the function Load_Instance(<path_2_instance>) to load an already created CODARFE model.')

    self.__log_transform = None
    self.__min_target_log_transformed = None
    self.__max_target_log_transformed = None
    self.results = None
    self.score_best_model = None
    self.selected_taxa = None
    self.__model = None
    self.__n_max_iter_huber = None
    
    self.__correlation_list = {}

  def get_path2metadata(self):
    return self.__path2MetaData
  
  def __Read_Data(self,path2Data,path2metadata,target_column_name):

    extension = path2Data.split('.')[-1]
    if extension == 'csv':
        data = pd.read_csv(path2Data,encoding='latin1')
        data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'tsv':
        data = pd.read_csv(path2Data,sep='\t',encoding='latin1')
        data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'biom':
        table = load_table(path2Data)
        data = table.to_dataframe()
    elif extension == 'qza':
        output_directory =  '/'.join(path2Data.split('/')[:-1])+'/QZA_EXTRACT_CODARFE_TEMP/'
        # Openning the .qza file as an zip file
        with zipfile.ZipFile(path2Data, 'r') as zip_ref:
            # extracting all data to the output directory
            zip_ref.extractall(output_directory)
        # Getting the biom path file
        biompath = output_directory+os.listdir(output_directory)[0]+'/data/'
        biompath += [f for f in os.listdir(biompath) if f[-5:]=='.biom'][0]
        table = load_table(biompath)# Read the biom file
        data = table.to_dataframe() # Tranform it to a pandas dataframe

        shutil.rmtree(output_directory) # remove the pathTree created

    extension = path2metadata.split('.')[-1]
    if extension == 'csv':
        metadata = pd.read_csv(path2metadata,encoding='latin1')
        metadata.set_index(list(metadata.columns)[0],inplace=True)
    elif extension == 'tsv':
        metadata = pd.read_csv(path2metadata,sep='\t',encoding='latin1')
        metadata.set_index(list(metadata.columns)[0],inplace=True)
    self.metadata = metadata
    totTotal = len(metadata)
    if target_column_name not in metadata.columns:
      print("The Target is not present in the metadata table!")
      sys.exit(1)
    totNotNa = metadata[target_column_name].isna().sum()
    metadata.dropna(subset=[target_column_name],inplace=True)

    #metadata = metadata[target_column_name]
    #X = metadata.join(data)
    data = data.loc[metadata.index]
    y = metadata[target_column_name]

    if len(data) == 0:
      print('There is no relationship between the predictor ids and the metadata. Ascertain that the first column corresponds to the identifiers.')
      sys.exit(1)
    print('Total samples with the target variable: ',totTotal-totNotNa,'/',totTotal)

    return data,y,metadata

  def Save_Instance(self,path_out,name_append = ''):

    if type(self.data) == type(None):
      print('Nothing to save.\n\nPlease when creating the CODARFE instance provide full data information: <path_2_Data> , <path_2_MetaData> and <metaData_Target>')
      return

    obj = {'data':self.data,
           'target':self.target,
           'metaData_Target':self.metaData_Target,
           'metadata':self.metadata,
           'path2MetaData':self.__path2MetaData,
           'log_transform': self.__log_transform,
           'min_target_log_transformed': self.__min_target_log_transformed,
           'max_target_log_transformed': self.__max_target_log_transformed,
           'results': self.results,
           'score_best_model': self.score_best_model,
           'selected_taxa': self.selected_taxa,
           'model': self.__model,
           'n_max_iter_huber': self.__n_max_iter_huber,
           'correlation_list':self.__correlation_list}
    if self.__path2MetaData.split('/')[:-1] != []:
        path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'
    # if path_out == '':
    #   path_out = '/'.join(os.path.abspath(self.__path2MetaData).split('/')[:-1])+'/'
    if name_append!='':
      name_append = 'CODARFE_MODEL_'+name_append+'.foda'
    else:
      name_append = 'CODARFE_MODEL.foda'
    name = path_out+name_append
    with open(name,'wb') as f:
      pk.dump(obj,f)

    print('\n\nInstance saved at ',name+'\n\n')

  def Load_Instance(self,path2instance):
    with open(path2instance,'rb') as f:
      obj = pk.load(f)

    self.data = obj['data']
    self.target = obj['target']
    self.metaData_Target = obj['metaData_Target']
    self.metadata = obj['metadata']
    self.__path2MetaData = obj['path2MetaData']
    self.__log_transform = obj['log_transform']
    self.__min_target_log_transformed = obj['min_target_log_transformed']
    self.__max_target_log_transformed = obj['max_target_log_transformed']
    self.results = obj['results']
    self.score_best_model = obj['score_best_model']
    self.selected_taxa = obj['selected_taxa']
    self.__model = obj['model']
    self.__n_max_iter_huber = obj['n_max_iter_huber']
    self.__correlation_list = obj['correlation_list']

    print('\n\nInstance restored successfully!\n\n')


  def __removeLowVar(self):
    aux = self.data.copy()
    cols = aux.columns
    selector = VarianceThreshold(aux.var(axis=1).mean()/8)#8
    aux = selector.fit(aux)
    not_to_drop=list(cols[selector.get_support()])
    totRemoved = len(self.data.columns) - len(not_to_drop)
    print('\nA total of ',totRemoved,' taxa were removed due to very low variance\n')
    self.data =  self.data[not_to_drop]

  def __toAbunRel(self,data):
    return data.apply(lambda x: x/x.sum() if x.sum()!=0 else x,axis=1)

  def __calc_new_redimension(self,target):
    target = np.log2(target)# 1° transforma pra log2

    minimo = min(target)
    maximo = max(target)
    if self.__min_target_log_transformed == None or self.__max_target_log_transformed == None:
      self.__min_target_log_transformed = minimo
      self.__max_target_log_transformed = maximo

    new_min = 0  # novo valor mínimo desejado
    new_max = 100  # novo valor máximo desejado

    numeros_redimensionados = [(x - minimo) / (maximo - minimo) * (new_max - new_min) + new_min for x in target]
    return numeros_redimensionados

  def __calc_inverse_redimension(self,predictions):

    new_min = 0  # novo valor mínimo usado na transformação
    new_max = 100  # novo valor máximo usado na transformação

    minimo = self.__min_target_log_transformed
    maximo = self.__max_target_log_transformed

    numeros_restaurados = [(x - new_min) / (new_max - new_min) * (maximo - minimo) + minimo for x in predictions]
    numeros_restaurados_log_inverse = np.power(2, numeros_restaurados)
    return numeros_restaurados_log_inverse

  def __toCLR(self,df): # Transform to CLr
    # aux = df.copy()
    # cols = aux.columns
    # aux+=0.0001 # Pseudo count
    # aux = clr(aux)
    # aux = pd.DataFrame(data=aux,columns=cols)
    # return aux

    aux = df.copy()
    aux+=0.0001 # Pseudo count
    aux = aux.apply(lambda x: np.log(x/np.exp(np.log(x).mean())),axis=1)
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

  def __superRFE(self,method,n_cols_2_remove,n_Kfold_CV):
    # Define o total de atributos a serem removidos por rodada
    n_cols_2_remove = max([int(len(self.data.columns)*n_cols_2_remove),1])

    # Define X inicial como descritores
    X_train = self.data
    tot2display = len(list(X_train.columns))
    # Define variável alvo
    if self.__log_transform:
      y_train = np.array(self.__calc_new_redimension(self.target))
    else:
      y_train = self.target

    # Inicializa tabela de resultados
    resultTable = pd.DataFrame(columns=['Atributos','R² adj','F-statistic','BIC','MSE-CV'])
    percentagedisplay = round(100 - (len(list(X_train.columns))/tot2display)*100)
    while len(X_train.columns) - n_cols_2_remove > n_cols_2_remove or len(X_train.columns) > 1:
      #print('1')
      X = self.__toCLR(X_train)
      #print('2')
      method.fit(X,y_train)
      #print('3')
      pred = method.predict(X)
      #print('4')
      Fprob = self.__calc_f_prob_centeredv2(pred,y_train,X)
      #print('5')
      r2 = self.__calc_r_squared(pred,y_train)
      #print('6')
      r2adj = self.__calc_rsquared_adj(X,r2)
      #print('7')
      BIC = self.__calc_bic(pred,y_train,X)

      msecv_results = []
      #Adicionando etapa de validação cruzada
      #print('8')
      kf = KFold(n_splits=n_Kfold_CV,shuffle=True,random_state=42)

      for train, test in kf.split(X):

        model = method.fit(X.iloc[train],y_train[train])

        predcv = model.predict(X.iloc[test])

        msecv_results.append(np.mean([(t-p)**2 for t,p in zip(y_train[test],predcv)])**(1/2))

      msecv = np.mean(msecv_results)

      # caso consiga calcular um p-valor
      if not math.isnan(Fprob) and r2adj < 1.0:
        # Cria linha com: Atributos separados por ';', R² ajustado, estatistica F e estatistica BIC
        newline = pd.DataFrame.from_records([{'Atributos':'@'.join(list(X.columns)),'R² adj':float(r2adj),'F-statistic':float(Fprob),'BIC':float(BIC),'MSE-CV':float(msecv)}])

        # Adiciona linha na tabela de resultado
        resultTable = pd.concat([resultTable,newline])

      # Cria tabela de dados dos atributos
      atr = pd.DataFrame(data = [[xx,w] for xx,w in zip(X.columns,method.coef_)],columns = ['Atributos','coef'])

      # Transforma coluna de coeficiente p/ float
      atr = atr.astype({'coef':float})

      # Remove colunas com coeficientes iguais ou menores que 0 / Acelera o rfe
      atr = atr[atr.coef != 0]# Remove zeros
      atr.coef = atr.coef.abs()#Transforma em abs para remover apenas os X% mais perto de zero

      # Ordena de forma decrescente pelo coeficiente
      atr = atr.sort_values(by=['coef'],ascending=False)

      # Cria lista de atributos selecionados
      atributos = list(atr.Atributos)

      # Remove espaços em brancos inseridos pela tabela de atributos
      atributos = [x.strip() for x in atributos]

      # Remove n_cols_2_remove menos relevantes
      atributos = atributos[:-n_cols_2_remove]

      # Remove atributos n selecionados
      X_train = X_train[atributos]

      # Calculo da % para mostrar na tela
      percentagedisplay = round(100 - (len(list(X_train.columns))/tot2display)*100)
      #print(percentagedisplay,'% done...\n')
    #Remove possiveis nan
    resultTable.dropna(inplace=True)
    # Retorna a tabela com os resultados
    return resultTable

  def __scoreAndSelection(self,resultTable,weightR2,weightProbF,weightBIC,weightRMSE):
    # Cria cópia da tabela original
    df_aux = resultTable.copy()
    df_aux.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_aux.dropna(inplace=True)

    # Normaliza o R² ajustado
    df_aux['R² adj'] = MinMaxScaler().fit_transform(np.array(df_aux['R² adj']).reshape(-1,1))

    # Aplica a transformação -log10(f) e então normaliza a estatistica f
    df_aux['F-statistic'] = [-math.log10(x) if x !=0 else sys.float_info.max for x in df_aux['F-statistic']]
    df_aux['F-statistic'] = MinMaxScaler().fit_transform(np.array(df_aux['F-statistic']).reshape(-1,1))

    # Normaliza e então inverte a estatistica BIC (Quanto menor menor)
    df_aux['BIC'] = MinMaxScaler().fit_transform(np.array(df_aux['BIC']).reshape(-1,1))
    df_aux['BIC'] = [np.clip(1-x,0,1) for x in df_aux['BIC']]

    # Normaliza 'MSE-CV' e inverte a estatistica MSE-cv
    df_aux['MSE-CV'] = MinMaxScaler().fit_transform(np.array(df_aux['MSE-CV']).reshape(-1,1))
    df_aux['MSE-CV'] = [np.clip(1-x,0,1) for x in df_aux['MSE-CV']]

    # Cria coluna de Score
    df_aux['Score'] = [(r*weightR2)+(f*weightProbF)+(b*weightBIC)+(m*weightRMSE) for r,f,b,m in zip(df_aux['R² adj'],df_aux['F-statistic'],df_aux['BIC'],df_aux['MSE-CV'])]

    # Encontra indice de maior Score
    indexSelected = list(df_aux.Score).index(max(list(df_aux.Score)))

    # Seleciona atributos
    selected = df_aux.iloc[indexSelected].Atributos.split('@')

    self.selected_taxa = selected # salva os atributos selecionados

    retEstatisticas = list(resultTable.iloc[indexSelected][['R² adj','F-statistic','BIC','MSE-CV']])

    self.results = {'R² adj':retEstatisticas[0],
                    'F-statistic':retEstatisticas[1],
                    'BIC':retEstatisticas[2],
                    'MSE-CV':retEstatisticas[3]} # Salva as estatisticas

    retScore = df_aux.iloc[indexSelected].Score

    self.score_best_model = retScore # Salva a pontuação do melhor modelo

  def __write_results(self,path_out,name_append):
    # adiciona '/' caso n tneha
    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if self.__path2MetaData.split('/')[:-1] != []:
        path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'
    # adiciona '_' caso n tenha
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append

    path2write = path_out+'CODARFE_RESULTS' +name_append+'.txt'
    print('Writing results at ',path2write,'\n')

    with open(path2write,'w') as f:
      f.write('Results: \n\n')
      f.write('R² adj -> '+     str(self.results['R² adj'])+'\n')
      f.write('F-statistic -> '+str(self.results['F-statistic'])+'\n')
      f.write('BIC -> '+        str(self.results['BIC'])+'\n')
      f.write('MSE-CV -> '+     str(self.results['MSE-CV'])+'\n')
      f.write('Total of taxa selected -> '+str(len(self.selected_taxa))+'. This value corresponds to '+str((len(self.selected_taxa)/len(self.data.columns))*100)+'% of the total observed\n\n')
      f.write('Selected taxa: \n\n')
      f.write(','.join(self.selected_taxa))
    
    method = HuberRegressor(epsilon = 2.0,alpha = 0.0003, max_iter = self.__n_max_iter_huber)
    X = self.data[self.selected_taxa]
    X = self.__toCLR(X)
    y = self.target
    resp = method.fit(X,y)

    dfaux = pd.DataFrame(data={'Predictors':resp.feature_names_in_,'coefs':resp.coef_})
    dfaux.sort_values(by='coefs',ascending=False,inplace=True,ignore_index=True)
    
    path2writeCoefs = path_out+'Predictors_weigths' +name_append+'.csv'
    print("Writing predictors weights at ",path2writeCoefs,'\n')
    dfaux.to_csv(path2writeCoefs,index=False)
  
  def __DefineModel(self):

      self.__model = RandomForestRegressor(n_estimators = 160, criterion = 'poisson',random_state=42)
      X = self.data[self.selected_taxa]
      X = self.__toCLR(X)

      if np.std(self.target)/np.mean(self.target)>0.2:# Caso varie muitas vezes a média (ruido)
        targetLogTransformed = self.__calc_new_redimension(self.target) # Aplica transformação no alvo
        self.__model.fit(X,targetLogTransformed) # Treina com o alvo transformado
        self.__log_transform = True # Define flag de transformação
        #print('The target was log transformed!')
      else:
        self.__model.fit(X,self.target) # Treina um segundo modelo com o alvo como é
        self.__log_transform = False # Define flag de transformação

  def __checkModelParams(self,
                         write_results,
                         path_out,
                         name_append,
                         rLowVar,
                         applyAbunRel,
                         percentage_cols_2_remove,
                         n_Kfold_CV,
                         weightR2,
                         weightProbF,
                         weightBIC,
                         weightRMSE,
                         n_max_iter_huber):
    if type(write_results) != bool:
      print("\n\nWARNING!\n\nwrite_results MUST be True or False")
      return False

    if type(rLowVar) != bool:
      print("\n\nWARNING!\n\nrLowVar MUST be True or False")
      return False

    if type(applyAbunRel) != bool:
      print("\n\nWARNING!\n\napplyAbunRel MUST be True or False")
      return False

    if type(path_out) != str:
      print("\n\nWARNING!\n\npath_out MUST be a String")
      return False
    
    elif write_results and path_out!='' and not os.path.exists(path_out):
      print('\n\nWARNING!\n\nThe path out does not exists!')
      return False


    if type(name_append) != str:
      print("\n\nWARNING!\n\nname_append MUST be a String")
      return False

    try:
      percentage_cols_2_remove = int(percentage_cols_2_remove)
    except:
      print('\n\nWARNING!\n\npercentage_cols_2_remove MUST be a integer')
      return False

    if percentage_cols_2_remove < 1 or percentage_cols_2_remove > 99:
      print('\n\nWARNING!\n\npercentage_cols_2_remove MUST be between 1 and 99')
      return False

    try:
      n_Kfold_CV = int(n_Kfold_CV)
    except:
      print('\n\nWARNING!\n\nn_Kfold_CV MUST be a integer')
      return False

    if n_Kfold_CV < 2 or n_Kfold_CV > 100:
      print('\n\nWARNING!\n\nn_Kfold_CV MUST be between 2 and 99')
      return False

    try:
      n_max_iter_huber = int(n_max_iter_huber)
    except:
      print('\n\nWARNING!\n\nn_max_iter_huber MUST be a integer')
      return False

    if n_max_iter_huber < 2:
      print('\n\nWARNING!\n\nn_max_iter_huber MUST be greater than 2')
      return False

    try:
      weightR2 = float(weightR2)
    except:
      print('\n\nWARNING!\n\nweightR2 MUST be a float')
      return False

    if weightR2 < 0:
      print('\n\nWARNING!\n\nweightR2 MUST be greater or equal to 0')
      return False

    try:
      weightProbF = float(weightProbF)
    except:
      print('\n\nWARNING!\n\nweightProbF MUST be a float')
      return False

    if weightProbF < 0:
      print('\n\nWARNING!\n\nweightProbF MUST be greater or equal to 0')
      return False

    try:
      weightBIC = float(weightBIC)
    except:
      print('\n\nWARNING!\n\nweightBIC MUST be a float')
      return False

    if weightBIC < 0:
      print('\n\nWARNING!\n\nweightBIC MUST be greater or equal to 0')
      return False

    try:
      weightRMSE = float(weightRMSE)
    except:
      print('\n\nWARNING!\n\nweightRMSE MUST be a float')
      return False

    if weightRMSE < 0:
      print('\n\nWARNING!\n\nweightRMSE MUST be greater or equal to 0')
      return False

    return True

  def CreateModel(self,
                  write_results: bool =True,
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


    if type(self.data) == type(None):
      print('No data was provided!\nPlease make sure to provide complete information or use the Load_Instance(<path_2_instance>) function to load an already created CODARFE model')
      return None
    
    print('\n\nChecking model parameters...',end='')
    if not self.__checkModelParams(write_results,path_out,name_append,rLowVar,applyAbunRel,percentage_cols_2_remove,n_Kfold_CV,weightR2,weightProbF,weightBIC,weightRMSE,n_max_iter_huber):
      print('\nPlease, Re-run this function with the correct parameters.')
      return None
    
    if write_results and path_out=='':
      nameout = '/'.join(self.__path2MetaData.split('/')[:-1])
      if nameout == '':
        nameout = 'this same folder'
      print('\n\nWARNING!\n\nThe path out was not provided!\nThe model will be written at ',nameout)
    
    print('OK')

    n_cols_2_remove = percentage_cols_2_remove/100
    self.__n_max_iter_huber = n_max_iter_huber # Define o numero de iterações utilziado pelo huber

    if rLowVar:
      #Remove baixa variância
      self.__removeLowVar()

    if applyAbunRel:
      #transforma em abundância relativa
      self.data = self.__toAbunRel(self.data)
      print('\n\nRelative abundance finished!\n\n')

    method = HuberRegressor(epsilon = 2.0,alpha = 0.0003, max_iter = n_max_iter_huber)

    print("\n\nSTARTING RFE!\n\n")

    # Remove iterativamente atributos enquanto cria vários modelos
    resultTable = self.__superRFE(method,n_cols_2_remove,n_Kfold_CV)
    print("\n\nFinished!\n\nSelecting best predictors...",end='')
    if len(resultTable)>0:
      # Calcula pontuação e seleciona o melhor modelo
      self.__scoreAndSelection(resultTable,weightR2,weightProbF,weightBIC,weightRMSE)
      print("DONE! \n\n")
      self.__DefineModel()

      print('\nModel created!\n\n')
      print('Results: \n\n')
      print('R² adj -> ',     self.results['R² adj'])
      print('F-statistic -> ',self.results['F-statistic'])
      print('BIC -> ',        self.results['BIC'])
      print('MSE-CV -> ',     self.results['MSE-CV'])
      print('Total of taxa selected -> ',len(self.selected_taxa),'. This value corresponds to ',(len(self.selected_taxa)/self.__totalPredictorsInDatabase)*100,'% of the total.\n')

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

  def __CreateCorrelationImputer(self):
    threshold = 0.6 # Considered as strong correlation
    aux = self.__toCLR(self.data) # Remove composicionalidade usando CLR nos dados originais

    for selected in self.selected_taxa: # Para cada taxa selecionada
      self.__correlation_list[selected] = [] # Cria instancia para esta taxa selecionada
      for taxa in aux.columns: # Verifica correlação com todas as outras
        if taxa != selected: # n comparar consigo mesmo
          corr = self.__pairwise_correlation(np.array(aux[selected]),np.array(aux[taxa]))# Calcula a correlação de forma rapida
          if corr >= threshold: # Somenta adiciona caso seja fortemente correlacionada
            self.__correlation_list[selected].append({'taxa':taxa,'corr':corr}) # Adiciona taxa correlacionada
      self.__correlation_list[selected].sort(reverse=True,key = lambda x: x['corr']) # Ordena pela correlação

  def __Read_new_Data(self,path2Data):

    extension = path2Data.split('.')[-1]
    if extension == 'csv':
        data = pd.read_csv(path2Data,encoding='latin1')
        data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'tsv':
        data = pd.read_csv(path2Data,sep='\t',encoding='latin1')
        data.set_index(list(data.columns)[0],inplace=True)
    elif extension == 'biom':
        table = load_table(path2Data)
        data = table.to_dataframe()
    elif extension == 'qza':
        output_directory =  '/'.join(path2Data.split('/')[:-1])+'/QZA_EXTRACT_CODARFE_TEMP/'
        # Openning the .qza file as an zip file
        with zipfile.ZipFile(path2Data, 'r') as zip_ref:
            # extracting all data to the output directory
            zip_ref.extractall(output_directory)
        # Getting the biom path file
        biompath = output_directory+os.listdir(output_directory)[0]+'/data/'
        biompath += [f for f in os.listdir(biompath) if f[-5:]=='.biom'][0]
        table = load_table(biompath)# Read the biom file
        data = table.to_dataframe() # Tranform it to a pandas dataframe

        shutil.rmtree(output_directory) # remove the pathTree created
    return data

  def Predict(self,
              path2newdata,
              applyAbunRel = True,
              writeResults = True,
              path_out = '',
              name_append = ''
              ):
    if self.__model == None:
      print('No model created, please run the function CreateModel first.')
      return None

    if writeResults and path_out != '' and not os.path.exists(path_out):
      print('\n\nThe path out does not exists!\nPlease try again with the correct path or let it blank to write in the same path as the metadata')
      return None

    new = self.__Read_new_Data(path2newdata)
    newindex = new.index

    if self.__correlation_list == {}:
      print('\n\nCreating correlation list for imputation method. It may take a few minutes depending on the size of the original dataset, but it will be create only once.\n\n')
      self.__CreateCorrelationImputer()
      print('Correlation list created!\n\n')
    else:
      print('\n\nCorrelation list was alread created. Using the existent one!\n\n')

    data2predict = pd.DataFrame() # Cria um dataframe para colocar apenas os dados selecionados
    totalNotFound = 0
    for selected in self.selected_taxa: # Para cada taxa selecionada
      if selected in new.columns: # Caso exista no novo conjunto
        data2predict[selected] = new[selected] # Adiciona o valor do novo conjunto no df de previsão
      else: # Senão
        found = False # Flag que indica se encontrou substitudo
        for correlated_2_selected in self.__correlation_list[selected]: # Para cada taxa correlacionada com a que não existe
          if correlated_2_selected['taxa'] in new.columns: # Caso encontre um substituto
            data2predict[selected] = new[correlated_2_selected['taxa']] # Coloca ele no lugar do que n existe
            found = True # Seta flag
            break
        if not found:
          data2predict[selected] = 0 # Caso não encontra retorna zero
          print('Warning! Taxa ',selected,' was not found and have no correlations! It may affect the model accuracy')
          totalNotFound+=1

    if totalNotFound >= len(self.selected_taxa)*0.75:
      print('The new samples has less than 25% of selected taxa. The model will not be able to predict it.')
      return None,totalNotFound

    # print('totalNotFound -> ',totalNotFound,'Total taxa ->',len(self.selected_taxa))

    # print('\n\n\ncomo veio ao mundo \n\n\n')
    # print(data2predict.head())
    data2predict = data2predict.fillna(0)

    if applyAbunRel:
      data2predict = self.__toAbunRel(data2predict) # Transforma em abundancia relativa

    # print('\n\n\nabundancia relativa \n\n\n')
    # print(data2predict.head())

    data2predict = self.__toCLR(data2predict) # Transforma para CLR

    # print('\n\n\nCLR \n\n\n')
    # print(data2predict.head())

    resp = self.__model.predict(data2predict)

    if self.__log_transform: # Caso o modelo tenha sido treinado nos dados log transformados
      resp = self.__calc_inverse_redimension(resp)#,totalNotFound # Retorna os valores restaurados ao original
    # else:
    #   return resp,totalNotFound # Senão retorna como esta

    if writeResults:
      
      if path_out != '':
        if path_out[-1]!= '/':
          path_out+='/'
      else:
        if type(self.__path2MetaData) != type(None):
          if self.__path2MetaData.split('/')[:-1] != []:
            path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'

      if name_append != '':
        name_append = '_'+name_append
      filename = path_out+'Prediction'+name_append+'.csv'
      pd.DataFrame(data = resp,columns = ['Prediction'],index=newindex).to_csv(filename)

    return resp,totalNotFound


  def Plot_Correlation(self,path_out='',name_append=''):
    if self.__model == None:
      print('No model created, please run the function CreateModel first.')
      return None

    if path_out != '' and not os.path.exists(path_out):
      print('\n\nThe path out does not exists!\nPlease try again with the correct path or let it blank to write in the same path as the metadata')
      return None

    # Build a rectangle in axes coords
    left, width = .15, .75
    bottom, height = .15, .75
    right = left + width
    top = bottom + height
    y = self.target
    X = self.data[self.selected_taxa]
    X = self.__toCLR(X)
    pred = self.__model.predict(X)

    if self.__log_transform: # Caso tenha aprendido com valores transformados
      pred = self.__calc_inverse_redimension(pred) # Destransforma-os
    plt.figure()
    plt.clf()
    ax = plt.gca()

    corr, what = pearsonr(y, pred)

    #Plota os pontos previsto por esperado
    plt.plot(pred, y, 'o')

    rawData = pd.DataFrame({'Real':y,'Prediction':pred})

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
      if self.__path2MetaData.split('/')[:-1] != []:
        path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'
    # adiciona '_' caso n tenha
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append

    filenameimg = path_out+'Correlation'+name_append+'.png'
    filenameraw = path_out+'Correlation_raw_data'+name_append+'.csv'
    print('\nSaving the image at ',filenameimg,'\n')
    plt.savefig(filenameimg, dpi=600, bbox_inches='tight')
    print('\nSaving the raw data at ',filenameraw,'\n')
    rawData.to_csv(filenameraw)

  def __checkHoldOutParams(self,n_repetitions,test_size,path_out,name_append):
    if type(n_repetitions) != int:
      print('\n\nWARNING!\n\nn_repetitions MUST be a integer!')
      return False
    elif n_repetitions < 2:
      print('\n\nWARNING!\n\nn_repetitions MUST be at least 2!')
      return False
    if type(test_size) != int:
      print('\n\nWARNING!\n\ntest_size MUST be a integer!')
      return False
    elif test_size<=0:
      print('\n\nWARNING!\n\ntest_size MUST be greater than 0!')
      return False
    if path_out != '' and not os.path.exists(path_out):
      print('\n\nWARNING!\n\nThe path out does not exists!\nPlease try again with the correct path or let it blank to write in the same path as the metadata')
      return False
    return True

  def Plot_HoldOut_Validation(self,
                              n_repetitions = 100,
                              test_size=20,
                              path_out='',
                              name_append=''):
    if self.__model == None:
      print('No model created, please run the function CreateModel first.')
      return None
    
    if not self.__checkHoldOutParams(n_repetitions,test_size,path_out,name_append):
      return None
    test_size = test_size/100
    method = RandomForestRegressor(n_estimators = 160, criterion = 'poisson',random_state=42)
    X = self.data[self.selected_taxa]
    X = self.__toCLR(X)
    y = self.target
    maes = []
    for i in range(n_repetitions):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) # divide em treino e teste

      if self.__log_transform: # Caso tenha aprendido originalmente com valores transformados
        method.fit(X_train,self.__calc_new_redimension(y_train)) # Re-treina com os valores transformados
        y_pred = method.predict(X_test) # Realiza a predição
        y_pred = self.__calc_inverse_redimension(y_pred) # Destransforma-os
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
      if self.__path2MetaData.split('/')[:-1] != []:
        path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'
    # adiciona '_' caso n tenha
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append

    filename = path_out+'HoldOut_Validation'+name_append+'.png'

    print('\nSaving the image at ',filename)
    plt.savefig(filename, dpi=600, bbox_inches='tight')


  def Plot_Relevant_Predictors(self,
                         n_max_features=100,
                         path_out='',
                         name_append=''):

    if self.__model == None:
      print('No model created, please run the function CreateModel first.')
      return None
    if type(n_max_features) != int:
      print('\n\nWARNING!\n\nn_max_features MUST be a integer!')
      return None
    elif n_max_features<2:
      print('\n\nWARNING!\n\nn_max_features MUST be at least 2!')
      return None
    if path_out != '' and not os.path.exists(path_out):
      print('\n\nWARNING!\n\nThe path out does not exists!\nPlease try again with the correct path or let it blank to write in the same path as the metadata')
      return None
    method = HuberRegressor(epsilon = 2.0,alpha = 0.0003, max_iter = self.__n_max_iter_huber)
    X = self.data[self.selected_taxa]
    X = self.__toCLR(X)
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
    
    rawData = dfaux.copy()
    
    ax.set_title('Strength of relevant predictors',fontweight='bold')
    ax.set(ylabel="Predictor name",
          xlabel="Coefficient weight")
    sns.despine(left=True, bottom=True)

    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if self.__path2MetaData.split('/')[:-1] != []:
        path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'
    # adiciona '_' caso n tenha
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append
  
    filenameimg = path_out+'Relevant_Predictors'+name_append+'.png'
    filenameraw = path_out+'Relevant_Predictors_raw_data'+name_append+'.csv'
    print('\nSaving the image at ',filenameimg,'\n')
    plt.savefig(filenameimg, dpi=600, bbox_inches='tight')
    print('\nSaving the raw data at ',filenameraw,'\n')
    rawData.to_csv(filenameraw)


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

  def Plot_Heatmap(self,
                   path_out='',
                   name_append=''):

    if self.__model == None:
      print('No model created, please run the function CreateModel first.')
      return None
    if path_out != '' and not os.path.exists(path_out):
      print('\n\nWARNING!\n\nThe path out does not exists!\nPlease try again with the correct path or let it blank to write in the same path as the metadata')
      return None
    # Pega o dataframe original porem apenas o que foi selecioando
    selected_features = self.data[self.selected_taxa]

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
    HeatMap_raw_data = pd.DataFrame(columns=y,index=bacs,data=bac_clr)

    im, cbar = self.__heatmap(bac_clr, bacs, y, ax=ax, cmap="RdYlBu",norm = norm, cbarlabel="Center-Log-Ratio")

    fig.tight_layout()


    if path_out != '':
      if path_out[-1]!= '/':
        path_out+='/'
    else:
      if self.__path2MetaData.split('/')[:-1] != []:
        path_out = '/'.join(self.__path2MetaData.split('/')[:-1])+'/'
    # adiciona '_' caso n tenha
    if name_append != '':
      if name_append[0]!= '_':
        name_append = '_'+name_append
  
    filename = path_out+'HeatMap'+name_append+'.png'
    filenameraw = path_out+'HeatMap_raw_data'+name_append+'.csv'
    print('\nSaving the image at ',filename)

    plt.savefig(filename,  bbox_inches='tight') #dpi=600, Removed because the image is huge! 
    print('\nSaving the Heat Map raw data at ',filenameraw)
    HeatMap_raw_data.to_csv(filenameraw)

def getArguments():
  argumentList = sys.argv[1:]
  #print(argumentList)
  if '-h' in argumentList or '--Help' in argumentList:
    print("########## NOTE ##########")
    print("If you DO NOT PROVIDE a path out for an result (such as model or graphics) it will be created at the original metadata\'s folder.")
    print("########## Help ##########\n")
    print('-h --Help: Display this Help.')
    print('-nt --NumberOfThreads: Define the maximum number of threads. Default = cpu_counts -1')
    print('-d --Data: The name of the counting table data file.')
    print('-m --Metadata: The name of the target\'s metadata file.')
    print('-t --Target: The name of the metadata column that the Target variable corresponds to.')
    print('-o --Output: The FOLDER\'s path, where ALL files will be saved. Default = metadata\'s folder')
    print('-na --NameAppend: To add a name to the model and results filenames. Default = None')
    print('-p --Predict: The name of the data file containing new samples that the model will predict.')
    print('\t-po --PredictionOutput: The path that is used when writing the prediction. Default = metadata\'s folder')
    print('\t-pna --PredictionNameAppend: The name that is added at the end of the predictions filename. Default = None')
    print('\t-pra --ApplyRAforPrediction: Apply the transformation for relative abundance to the new samples. (Set as False/f/F if your data is already in relative abundance). Default = True')
    print('-sm --SelectMetadata: Select the metadata based on RDA with the data used.')
    print('\t-smnm --SelectMetadataNewMeta: The path to a new metadata file to be used instead of the used to create the model.')
    print('\t-smpo --SelectMetadataOutput: The output path to save the metadata selected. Default = metadata\'s folder')
    print('\t-smna --SelectMetadataNameAppend: The name to append in the end of the filename to save the metadata selected. Default = None')
    print('-l --LoadInstance: The name of the instance file for an existing CODARFE model (.foda file)')
    print('\tSince it is saved in the Instance file, it is not necessary to use -d, -m, or -t when using -l.')
    print('\n---------- MODEL PARAMETERS ----------\n')
    print('-rlv --RemoveLowVar: Apply the low variance columns elimination. (For small datasets (less than 300 columns), set as False/f/F). Default = True')
    print('-ra --RelativeAbundance: Apply the transformation for relative abundance. (Set as False/f/F if your data is already in relative abundance). Default = True')
    print('-cr --PercentageColsToRemove: The number of columns to eliminate in each RFE iteration (for example, use 1 for 1%). Default = 1')
    print('-nf --NumberOfFolds: In the Cross-validation step, the number of folds. Default = 10')
    print('-hb --HuberRegression: The Huber regression\'s maximum number of iterations. Default = 100')
    print('-r2 --R2weight: Alter the R2\'s default weight in the RFE score. Default = 1.0')
    print('-pf --ProbFweight: Alter the Prob(F)\'s default weight in the RFE score. Default = 0.5')
    print('-rmse --RMSEweight: Alter the RFE score\'s default weighting of the RMSE. Default = 1.5')
    print('-bic --BICweight: Alter the default weight of the BIC in the RFE score. Default = 1.0')
    print('\n---------- PLOTS ----------\n')
    print('-pc --PlotCorrelation: Create a correlation plot with the generalized Vs target and save it.')
    print('-pco --PlotCorrelationPathOut: The path that is used when saving the Correlation Plot. Default = metadata\'s folder')
    print('\t-pcna --PlotCorrelationNameAppend: The name to put to the end of the plot\'s filename. Default = None')
    print('-ho --PlotHoldOut: Create and save the Hold-Out validation box-plot with the MAE.')
    print('-hoo --PlotHoldOutPathOut: The path that is used when saving the Hold-out Plot. Default = metadata\'s folder')
    print('\t-horep --HoldOutRepetitions: The number of Hold-Out repetitions (represented by the number of dots in the box plot). Default = 100')
    print('\t-hots --HoldOutTestSize: The percentage of the dataset that is utilized in the test (for example, 20% uses 20). Default = 20')
    print('\t-hona --HoldOutNameAppend: The name to put to the end of the plot\'s filename. Default = None')
    print('-hm --PlotHeatMap: Create and save a Heat-Map of the selected predictors\' CLR transformed abundance.')
    print('-hmo --PlotHeatMapPathOut: The path that is used when saving the Heat Map. Default = metadata\'s folder')
    print('\t-hmna --PlotHeatMapNameAppend: The name to put to the end of the plot\'s filename. Default = None')
    print('-rp --PlotRelevantPredictors: Create and save a bar plot containing the top -rpmax predictors, as well as their strengths and directions of correlation to the target.')
    print('-rpo --PlotRelevantPredictorsPathOut: The path that is used when saving the Relevant Predictors Plot. Default = metadata\'s folder')
    print('\t-rpmax --RelevantPredictorMaximum: In the -rp, pick the maximum number of predictors to display. Default = 100')
    print('\t-rpna --RelevantPredictorNameAppend: The name to put to the end of the plot\'s filename. Default = None')
    sys.exit(1)

  
  argdic = {}
  previous = ''
  alonelist = ['-pc','--PlotCorrelation','-ho','--PlotHoldOut','-hm','--PlotHeatMap','-rp','--PlotRelevantPredictors']
  for item in argumentList:
    if item.startswith('-'):
      if previous.startswith('-'):
        print("Error in parameter ",previous)
        sys.exit(1)
      if item in alonelist:
        argdic[item] = ''
      else:
        previous = item 
    else:
      if not previous.startswith('-'):
        print('Error in parameter ',item)
        sys.exit(1)
      elif previous == '':
        print('Error in parameter ',item)
        sys.exit(1)
      else:
        argdic[previous] = item 
        previous = ''
  #print(argdic)
  return argdic

def checkArguments(argdic):
  argsList = list(argdic.keys())
  if '-nt' in argsList:
    try :
      argdic['-nt'] = int(argdic['-nt'])
      if argdic['-nt']<=0:
        print('Number of Threads MUST be positive non zero!')
        sys.exit(1)
    except:
      print('Number of Threads MUST be integer!')
      sys.exit(1)
    
      
  if '-o' in argsList and argdic['-o'] != '' and not os.path.exists(argdic['-o']):
    print('The output FOLDER does not exists!')
    sys.exit(1)

  if '-l' not in argsList:
    # Checking Data
    if '-d' not in argsList:
      print('Please, provide a database or load an instance.')
      sys.exit(1)
    else:
      if not os.path.exists(argdic['-d']):
        print('The path for the data does not exists!')
        sys.exit(1)
      else:
        if argdic['-d'].split('.')[-1] not in ['biom','qza','csv','tsv']:
          print('The data must be in one of this formats: csv, tsv, biom or qza.')
          sys.exit(1)
      
    # checking metadata
    if '-m' not in argsList:
      print('Please, provide a metadata table or load an instance.')
      sys.exit(1)
    else:
      if not os.path.exists(argdic['-m']):
        print('The path for the metadata does not exists!')
        sys.exit(1)
      else:
        if argdic['-m'].split('.')[-1] not in ['csv','tsv']:
          print('The data must be in one of this formats: csv or tsv.')
          sys.exit(1)
      
      if '-t' not in argsList:
        print('Please, provide a Target or load an instance.')
        sys.exit(1)
      else:
        if '-t' in argsList and argdic['-t'] == '':
          print('The Target MUST NOT be empty!')
          sys.exit(1)
  else:
    if '-l' in argsList and not os.path.exists(argdic['-l']):
      print('The Instance file does not exists!')
      sys.exit(1)

  tfdict = {'False':False,'F':False,'f':False,'True':True,'T':True,'t':True}
  if '-rlv' in argsList:
    if argdic['-rlv'] not in tfdict.keys():
      print('-rlv MUST be one of: [True, T, t, False, F, f]')
      sys.exit(1)
    else:
      argdic['-rlv'] = tfdict[argdic['-rlv']]
  
  if '-ra' in argsList:
    if argdic['-ra'] not in tfdict.keys():
      print('-ra MUST be one of: [True, T, t, False, F, f]')
      sys.exit(1)
    else:
      argdic['-ra'] = tfdict[argdic['-ra']]
  
  if '-pra' in argsList:
    if argdic['-pra'] not in tfdict.keys():
      print('-pra MUST be one of: [True, T, t, False, F, f]')
      sys.exit(1)
    else:
      argdic['-pra'] = tfdict[argdic['-pra']]

  if '-p' in argsList:
    if not os.path.exists(argdic['-p']):
      print('The new samples database filename does not exists!')
      sys.exit(1)
    elif argdic['-p'].split('.')[-1] not in ['biom','qza','csv','tsv']:
      print('The new samples database MUST be in one of this formats: csv, tsv, biom or qza.')
      sys.exit(1)



def reduceArgs(argdict):  
  reducerule = {'--Help':'-h','--Data':'-d','--Metadata':'-m','--Target':'-t','--Output':'-o','--Predict':'-p','--LoadInstance':'-l','--RemoveLowVar':'-rlv','--RelativeAbundance':'-ra','--R2weight':'-r2','--ProbFweight':'-pf','--RMSEweight':'-rmse','--BICweight':'-bic','--PlotCorrelation':'-pc','--PlotHoldOut':'-ho','--HoldOutRepetitions':'-horep','--HoldOutTestSize':'-hots','--PlotHeatMap':'-hm','--PlotRelevantPredictors':'-rp','--RelevantPredictorMaximum':'-rpmax','--PercentageColsToRemove':'-cr','--NumberOfFolds':'-nf','--ApplyRAforPrediction':'-pra','--PredictionOutput':'-po','--NameAppend':'-na','--PredictionNameAppend':'-pna','--PlotCorrelationNameAppend':'-pcna','--HoldOutNameAppend':'-hona','--PlotHeatMapNameAppend':'-hmna','--RelevantPredictorNameAppend':'-rpna','--NumberOfThreads':'-nt','--PlotRelevantPredictorsPathOut':'-rpo','--PlotHeatMapPathOut':'-hmo','--PlotHoldOutPathOut':'-hoo','--PlotCorrelationPathOut':'-pco'}
  reduced = {}
  for key in argdict.keys():
    if key in reducerule.keys():
      reduced[reducerule[key]] = argdict[key]
    else:
      reduced[key] = argdict[key]
  return reduced



def get_args():
  argdict = reduceArgs(getArguments())
  #print(argdict)
  checkArguments(argdict)
  
  argsList = list(argdict.keys())
  #print('argsList -> ',argsList)
  if '-o' in argsList:
    path_out = argdict['-o']
  else:
    path_out = ''
  return argsList ,argdict , path_out


def dotheRest(argsList ,argdict , path_out):
  if '-l' in argsList: 
    coda = CODARFE() 
    coda.Load_Instance(argdict['-l'])
    if not ('-o' in argsList) and not (os.path.exists(coda.get_path2metadata())):
      print('No output path was provided and the original metadata\'s folder does not exists anymore.')
      print('Please, provide a new output path to save the files')
      sys.exit(1)
  else:
    coda = CODARFE(path2Data=argdict['-d'],
                   path2MetaData=argdict['-m'],
                   metaData_Target=argdict['-t'])
  
  if type(coda.selected_taxa) == type(None):
    print('\n\nCREATING MODEL\n')
    write_results = True
    path_out = path_out
    name_append = argdict['-na'] if '-na' in list(argdict.keys()) else ''
    rLowVar =argdict['-rlv'] if '-rlv' in list(argdict.keys()) else True
    applyAbunRel = argdict['-ra'] if '-ra' in list(argdict.keys()) else True
    percentage_cols_2_remove = argdict['-cr'] if '-cr' in list(argdict.keys()) else 1
    n_Kfold_CV=argdict['-nf'] if '-nf' in list(argdict.keys()) else 10
    weightR2 =argdict['-r2'] if '-r2' in list(argdict.keys()) else 1.0
    weightProbF=argdict['-pf'] if '-pf' in list(argdict.keys()) else 0.5
    weightBIC=argdict['-bic'] if '-bic' in list(argdict.keys()) else 1.0
    weightRMSE=argdict['-rmse'] if '-rmse' in list(argdict.keys()) else 1.5
    n_max_iter_huber=argdict['-hb'] if '-hb' in list(argdict.keys()) else 100

    coda.CreateModel(write_results = write_results,
                    path_out = path_out,
                    name_append = name_append,
                    rLowVar = rLowVar,
                    applyAbunRel = applyAbunRel,
                    percentage_cols_2_remove = percentage_cols_2_remove,
                    n_Kfold_CV = n_Kfold_CV,
                    weightR2 = weightR2,
                    weightProbF = weightProbF,
                    weightBIC = weightBIC,
                    weightRMSE = weightRMSE,
                    n_max_iter_huber = n_max_iter_huber)
    
    coda.Save_Instance(path_out=path_out,
                      name_append = name_append)
  else:
    print('\n\nModel Already Created. Skipping model creation!\n\n')

  if '-p' in argsList:
    print('\n\nPREDICTING\n')
    coda.Predict(path2newdata = argdict['-p'] if '-p' in list(argdict.keys()) else '',
                applyAbunRel = argdict['-pra'] if '-pra' in list(argdict.keys()) else True,
                writeResults = True,
                path_out = argdict['-po'] if '-po' in list(argdict.keys()) else '',
                name_append = argdict['-pna'] if '-pna' in list(argdict.keys()) else ''
                )
    if '-l' in argsList:
      fullname = argdict['-l']
      name_append = fullname.split('CODARFE_MODEL')[-1].split('.')[0]
      if name_append != '' and name_append[0]=='_':
        name_append = name_append[1:]
      path_out = fullname.split('CODARFE_MODEL')[0]
  
    coda.Save_Instance(path_out    = path_out,
                       name_append = name_append)

  if '-pc' in argsList:
    print('\n\nPlotting correlation...')
    coda.Plot_Correlation(path_out = argdict['-pco'] if '-pco' in list(argdict.keys()) else path_out,
                          name_append = argdict['-pcna'] if '-pcna' in list(argdict.keys()) else '')
    print('Done!\n\n')

  if '-ho' in argsList:
    print('\n\nPlotting Hold out validation...')
    coda.Plot_HoldOut_Validation(n_repetitions = argdict['-horep'] if '-horep' in list(argdict.keys()) else 100,
                                 test_size = argdict['-hots'] if '-hots' in list(argdict.keys()) else 20,
                                 path_out = argdict['-hoo'] if '-hoo' in list(argdict.keys()) else path_out,
                                 name_append = argdict['-hona'] if '-hona' in list(argdict.keys()) else '')
    print('Done!\n\n')

  if '-hm' in argsList:
    print('\n\nPlotting Heat Map...')
    coda.Plot_Heatmap(path_out = argdict['-hmo'] if '-hmo' in list(argdict.keys()) else path_out,
                      name_append = argdict['-hmna'] if '-hmna' in list(argdict.keys()) else '')
    print('Done!\n\n')

  if '-rp' in argsList:
    print('\n\nPlotting Relevant Predictors...')
    coda.Plot_Relevant_Predictors(n_max_features = argdict['-rpmax'] if '-rpmax' in list(argdict.keys()) else 100,
                                  path_out = argdict['-rpo'] if '-rpo' in list(argdict.keys()) else path_out,
                                  name_append = argdict['-rpna'] if '-rpna' in list(argdict.keys()) else '')
    print('Done!\n\n')

#main()
import os
import sys

argsList ,argdict , path_out = get_args() # Get all arguments

if not (type(argsList) == type(None) or type(argdict) == type(None) or type(path_out) == type(None)):

  # Load th rest of the libraryes
  from threadpoolctl import threadpool_limits
  from biom import load_table
  import zipfile
  import shutil

  from skbio.stats.composition import clr
  from skbio.stats.ordination import rda
  import pandas as pd
  import numpy as np
  import scipy.stats as stats
  import math
  import operator
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.feature_selection import VarianceThreshold
  from sklearn.ensemble import RandomForestRegressor
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

  import warnings
  warnings.filterwarnings("ignore")
  # Check the number of threads
  
  limits = argdict['-nt'] if '-nt' in list(argdict.keys()) else os.cpu_count()-1
  
  with threadpool_limits(limits=limits, user_api='blas'):
    dotheRest(argsList ,argdict , path_out)


#with threadpool_limits(limits=2, user_api='blas')

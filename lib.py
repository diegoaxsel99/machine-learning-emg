# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:04:31 2020

@author: Matador
"""
#librerias que se emplearon

from six.moves import urllib #libreria para importar los datos desde el link
import zipfile #libreria para descomprimir los archivos
import pandas as pd #libreria para manejar los datos en forma de sheet como excel
import numpy as np #libreria donde se encuentra las herramientas matematicas
import matplotlib.pyplot as plt # libreria para graficar la informacion
import joblib #libreria para guardar las variables de forma de pkl
from sklearn.model_selection import train_test_split, cross_validate,GridSearchCV #librerias ára realizar una validacion de los datos que enetran al modelo
from sklearn.neighbors import KNeighborsClassifier# modelo de k neighbors
import os # libreria para manejar los direcciones
import seaborn as sns # libreria para que la grafica tenga buena apariencia
import pywt # libreria donde se encuentra las herramientas de wavelet
from sklearn.decomposition import PCA # libreria para realizar la reduccion de dimensionalidad
from sklearn.tree import DecisionTreeClassifier # modelo de arbol de decision
from sklearn.ensemble import RandomForestClassifier# modelo de random forest
from sklearn.svm import SVC# modelo de support vector machine

# direccion donde se encuentra la base de datos
download_root = "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/"
# crear una path local para guardar la base de datos
path = os.path.join('database')
url = os.path.join(download_root,
                   "EMG_data_for_gestures-master.zip")

#funcion para descargar la base de datos en una carpeta
def fetch_database(path = path, url = url):
    print("importando los datos")
    # si la carpeta no existe se crea
    if not (os.path.isdir(path)):
        os.mkdir(path)
        
        # se crea una direccion donde se descarga el archivo zip
        base_path = os.path.join(path,'database.zip')
        # se descarga desde el link a la direccion anteriormente creada
        urllib.request.urlretrieve(url,base_path)
        # se descomprime los archivos 
        base_zip = zipfile.ZipFile(base_path)
        # se extraen toda la informacion
        base_zip.extractall(path = path)
        base_zip.close() 

#funcion para convertir todos los archivos de texto en solo archivo en forma de sheet de pandas
def transform_to_pkl():
    print("organizando la informacion")
    path0 = os.listdir()[0]
    path1 = os.listdir(path0)[1]
    
    dire = os.path.join(path0,path1)
    
    list_dire = os.listdir(dire)
    
    # se crea un dictionario donde se almacenara cada uno de los movientos
    movimiento={}
    # crear una lista donde se encuentra los nombres de las carpetas, para luego acceder a los
    # archivos txt
    
    
    for i in range(1,7):
        movimiento[str(i)] = [] # cada uno de los movimient4os enumerados como forma de lista
    
    for z in range(len(list_dire)-1):
        
        print(str(round((z/(len(list_dire)-1))*100) + 3)+"%")
        # se ingresa a la carpeta local
        local_path = os.path.join(dire,list_dire[z])
        # se crea una lista de los archivos dentro de la carpeta
        file = os.listdir(local_path)
        
        # un for para recorrer los archivos dentro de la carpeta
        for q in file:
            
            # se importa el archivo txt en forma de sheet de pandas
            df = pd.read_csv(os.path.join(local_path,q),
                             sep = '\t',
                             header = None)
            
            # se extraen los nombres de los archivos 
            names = df.iloc[0,:10].values
            # se reorganiza el sheet
            df = df.iloc[1:]
            # df.columns(names)
            
            
            values = df.values
            values_i = np.zeros(df.shape)
            # los valores se extraen de la sheet porque son str se transforman a int
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    
                    values_i[i,j] = float(values[i,j])
            
            # se vuelve a crear el sheet con los valores como int y columnas de los nombres
            df = pd.DataFrame(values_i,columns = names)
            
            aux_mov={}
            # se crea un auxliar debido a que en cada registro existen dos contracciones 
            # todo el proceso es para separalo y ingresarlo a movimientos
            for i in range(1,7):
                aux_mov[str(i)] = []
            
            for i in range(1,7):
            
                w = str(i)
                # se segmenta cada uno de los movimientos dentro del dictionario 
                # cada posicion del dictionario es una de las clases del movimiento con todos
                # sus canales en total son 8 canales que se refiere a los sensores
                
                aux_mov[w].append(df[df['class']==i].drop(['class'], axis = 1))
                 
            for i in range(len(aux_mov)):
                
                w = str(i + 1)
                df_local = aux_mov[w][0]
                # se extrae la columna de time que sirve como un indice
                # si el movimento de los indices es continuo o sea 1,2,3 es que la misma 
                # contraccion si hay salto es que es otra contraccion 40,500 en time
                x = df_local['time'].values
                df_local = df_local.drop(['time'], axis = 1)
                
                for j in range(len(x) - 1):
                    
                    if(x[j+1] - x[j] > 100):
                        # se decteta dicho salto y se parte el registro en dos y se guardan
                        movimiento[w].append(df_local.iloc[:j])
                        movimiento[w].append(df_local.iloc[j:])
    
    # se crea la carpeta datos si no existe 
    if not(os.path.isdir('datos')):
        os.mkdir('datos')
    # se guardan los datos segmentados en un archivo binarios 
    joblib.dump(movimiento,'datos/segmentados.pkl')
    
# metodo numerico para integrar una señal    
def integration(fx,n,b,con_ini):
    
    h = b/n
    t = np.linspace(0,b,n)
    
    gx = []
    
    gx.append(con_ini)
    
    for i in range(len(t) - 1):
        gx.append(gx[i] + h*fx[i])
        
    return gx

# funcion que integra las señales y crea un sheet con todas las señales integradas
def integrate_signals():
    
    print("integrado las señales")
    data = joblib.load('datos/segmentados.pkl')
    
    plt.close('all')
    # se recorre el sheet y se integra cada una de las señales
    for q in range(1,7):
        z = str(q)
        print(str(round((q/7)*100 + 14)) + "%")
        
        for c in range(len(data[z])):
            for w in range(1,9):
                
                x = data[z][c]['channel'+str(w)].values
                
                for i in range(len(x)):
                        
                    x[i] = abs(x[i])
                              
                gx = integration(x,len(x),3,0)
                
                data[z][c]['channel'+str(w)] = gx
    
    # se guarda la informacion 
    joblib.dump(data,'datos/señales integradas.pkl')

#creacion de la primera matrix de caracteristicas de las señales integradas
def matrix1():
    data = joblib.load('datos/señales integradas.pkl')
    # las medidas que se toman
    medidas = ["mean","var","std","max"]
    
    # creacion del dictionario se almaceran los datos
    data_ = {}
    for i in medidas:
        for j in range(1,9):
            w = i + str(j)
            data_[w] = []
            
    data_["target"] = []
    # se recorren los sheet y se calculan cada una de las medidas y se almacen en data_
    
    for q in range(1,7):        
        z = str(q)
        for j in range(len(data[z])):
            data_["target"].append(int(z))
            for i in range(1,9):
                x = data[z][j]['channel' + str(i)]
                
                data_['mean'+ str(i)].append(x.mean())
                data_['std'+ str(i)].append(x.std())
                data_['var'+ str(i)].append(x.var())
                data_['max'+ str(i)].append(x.max())
                
    # se crea el pandas       
    df = pd.DataFrame(data_,columns = list(data_.keys()))
    # se crea la carpeta donde se almacenaran los datos
    if not(os.path.isdir('features')):
        os.mkdir('features')
    
    # se guardan 
    joblib.dump(df,'features/matrix1.pkl')

# funcion que calcula la matrix de correlacion la muestra o no donde un df (DataFrame de pandas)
def plot_corr(df , name, show ,size = 20):
    
    cor_mat = df.corr()
    cols = df.columns
    sns.set(font_scale=1)
    
    plt.figure()
    
    heat_map = sns.heatmap(cor_mat,
                           cbar=True,
                           annot=True,
                           square=True,
                           fmt='.2f',
                           annot_kws={'size': 5},
                           yticklabels=cols,
                           xticklabels=cols)
    
    plt.title('Correlation Matrix - Heat Map')
    plt.show() 
    
    if not os.path.isdir('graficas'):
        os.mkdir('graficas')
    
    if not(show):
        plt.savefig('graficas/corr'+ name +'.png')
        plt.close('all')

# funcion que crea la segunda matrix de caracteristicas atravez de los coeficientes wavelet
def matrix2(wavelet):   
    path = os.path.join('datos','segmentados.pkl')
    data = joblib.load(path)
    
    
    data_ = {}
    # medidas que se tomaran
    medidas = ["mean","std","median"]
    # niveles de descomposicion
    niveles = ["cA","cD2","cD4"]
    
    # agragar lista cada una de las posiciones del dictionario
    for r in (niveles):
        for i in medidas:
            for j in range(1,9):
                w = r +  i +str(j)
                data_[w] = []
            
    data_['target'] = []
    
    # ingresar el df recorrerlo calcular los coeficientes y calcular las caracteristicas
    for q in range(1,7):
        z = str(q)
        for l in range(len(data[z])):
            data_['target'].append(q)
            for k in range(1,9):
                
                pot = data[z][l]['channel'+str(k)]
                
                
                
                cA ,cD1 = pywt.dwt(pot,wavelet)
                cA ,cD2 = pywt.dwt(cA,wavelet)    
                cA ,cD3 = pywt.dwt(cA,wavelet) 
                cA ,cD4 = pywt.dwt(cA,wavelet) 
                coeff = [cA,cD2,cD4]
                
                for r in range(len(niveles)):
                    
                    data_[niveles[r] + "mean"+str(k)].append(np.mean(coeff[r]))
                    data_[niveles[r] + "std"+str(k)].append(np.std(coeff[r]))
                    data_[niveles[r] + "median"+str(k)].append(np.median(coeff[r]))

    df = pd.DataFrame(data_,columns = list(data_.keys()))
    
    joblib.dump(df,'features/matrix2.pkl') 
# creacion de una funcion al ingresar df devuelva un df con la informacion 
# de precision y la desviacion de la misma
def knn(df):
    
    [f,c] = df.shape
    
    X = df.iloc[:,:c - 1]
    y = df.iloc[:,c - 1]
    
    knn = KNeighborsClassifier(n_neighbors = 5,
                               weights = "distance",
                               metric = "manhattan")
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 2)
    
    knn_score= cross_validate(knn,X_train,y_train,cv=10)
    knn_score2= cross_validate(knn,X_test,y_test,cv=10)
    
    precision = []
    std = []
    precision.append(np.mean(knn_score['test_score']))
    std.append(np.std(knn_score['test_score']))
    precision.append(np.mean(knn_score2['test_score']))
    std.append(np.std(knn_score2['test_score']))
    
    data_ = {"precision":precision,"desviacion":std}
    index = ["train","test"]
    
    return pd.DataFrame(data_, columns = list(data_.keys()), index = index)
# toma archivos que se encuentran en la carpeta de features y les aplica una reducion de dimesionalidad

def pca():
    
    
    path = os.path.join('features')
    list_dir = os.listdir(path)
    
    if not os.path.isdir("pca feature"):
        os.mkdir("pca feature")
    
    for i in list_dir:
        new_path = os.path.join(path,i)
        
        df = joblib.load(new_path)
        
        pca = PCA(n_components= 0.95)
        
        [f,c] = df.shape
        
        X = df.iloc[:,:c - 1] 
           
        X_new = pca.fit_transform(X)
        
        df_new = pd.DataFrame(X_new)
        
        df_new['target'] = df['target']
        
        joblib.dump(df_new,"pca feature/pca "+i)

# aplica gridseaarchcv para consegir los mejores parametros para un modelo
def grid(clf,df,parameters):
        
     [f,c] = df.shape
     
     X = df.iloc[:,:c - 1]
     y = df.iloc[:,c - 1]
     
     clf_ = GridSearchCV(clf, param_grid = parameters,
                         scoring = "accuracy",
                         cv = 10)
     
     clf_.fit(X,y)
     
     return clf_.best_params_
# modelo Decisiontree
def tree(df):
    
    [f,c] = df.shape
    
    X = df.iloc[:,:c - 1]
    y = df.iloc[:,c - 1]
    
    tree = DecisionTreeClassifier(random_state = 0)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 4)
    
    tree_score= cross_validate(tree,X_train,y_train,cv=10)
       
    return np.mean(tree_score['test_score']),np.std(tree_score['test_score'])

#modelo random forest
def random(df):
    
    [f,c] = df.shape
    
    X = df.iloc[:,:c - 1]
    y = df.iloc[:,c - 1]
    
    random = RandomForestClassifier(criterion = "gini",
                                  max_depth = 20,
                                  max_features = None,
                                  n_estimators = 20)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 2)
    
    random_score= cross_validate(random,X_train,y_train,cv=10)

    return np.mean(random_score['test_score']),np.std(random_score['test_score'])
# modelo suppor vector machine
def svc(df):
    
    [f,c] = df.shape
    
    X = df.iloc[:,:c - 1]
    y = df.iloc[:,c - 1]

    clf = SVC(decision_function_shape = "ovo")
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 2)
    
    svc_score= cross_validate(clf,X_train,y_train,cv=10)
    return np.mean(svc_score['test_score']),np.std(svc_score['test_score'])
 
# toma todas las posibble combinaciones de wavelets y las prueba sobre una matrix
def check_wavelets(direccion):
    
    wavelets = [
      'bior1.1',
     'bior1.3',
     'bior1.5',
     'bior2.2',
     'bior2.4',
     'bior2.6',
     'bior2.8',
     'bior3.1',
     'bior3.3',
     'bior3.5',
     'bior3.7',
     'bior3.9',
     'bior4.4',
     'bior5.5',
     'bior6.8',
     'coif1',
     'coif2',
     'coif3',
     'coif4',
     'coif5',
     'coif6',
     'coif7',
     'coif8',
     'coif9',
     'coif10',
     'coif11',
     'coif12',
     'coif13',
     'coif14',
     'coif15',
     'coif16',
     'coif17',
     'db1',
     'db2',
     'db3',
     'db4',
     'db5',
     'db6',
     'db7',
     'db8',
     'db9',
     'db10',
     'db11',
     'db12',
     'db13',
     'db14',
     'db15',
     'db16',
     'db17',
     'db18',
     'db19',
     'db20',
     'db21',
     'db22',
     'db23',
     'db24',
     'db25',
     'db26',
     'db27',
     'db28',
     'db29',
     'db30',
     'db31',
     'db32',
     'db33',
     'db34',
     'db35',
     'db36',
     'db37',
     'db38',
     'dmey',
     'haar',
     'rbio1.1',
     'rbio1.3',
     'rbio1.5',
     'rbio2.2',
     'rbio2.4',
     'rbio2.6',
     'rbio2.8',
     'rbio3.1',
     'rbio3.3',
     'rbio3.5',
     'rbio3.7',
     'rbio3.9',
     'rbio4.4',
     'rbio5.5',
     'rbio6.8',
     'sym2',
     'sym3',
     'sym4',
     'sym5',
     'sym6',
     'sym7',
     'sym8',
     'sym9',
     'sym10',
     'sym11',
     'sym12',
     'sym13',
     'sym14',
     'sym15',
     'sym16',
     'sym17',
     'sym18',
     'sym19',
     'sym20']
    
    _ = []
    for i in range(len(wavelets)):
        print(round((i/len(wavelets))*100))
        
        matrix2(wavelets[i])
        pca()
        
        df = joblib.load(direccion) 
        _.append(knn(df))
    
    sort = {"values":_,"wavelets":wavelets}
    
    df_ = pd.DataFrame(sort,columns =  list(sort.keys()))
    
    df_ = df_.sort_values('values', ascending = False) 
    
    return df_.iloc[0,0], df_.iloc[0,1]

# carga todos los datos
def init():
    
    path = os.path.join("database")
    
    if not os.path.isdir(path): 
        
        print("realizando todo los procesos")
        
        fetch_database(path = path, url = url)
        transform_to_pkl()
        integrate_signals()
        matrix1()
        matrix2('db7')
        pca()
        
    else:
        print("procesos ya realizados")

def convert_csv():
    
    path = os.path.join("features")
    folder = os.path.join("features csv")
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    listdir = os.listdir(path)
    
    k = 0
    for i in listdir:
        
        k = k + 1
        local_path = os.path.join(path,i)
        
        df = joblib.load(local_path)
        
        csv_path = os.path.join(folder, "matrix"+str(k)+".csv")
        
        df.to_csv(csv_path,index = False)
    
        
        
        
        
        
        
    
    
    
        
    
    
    
    
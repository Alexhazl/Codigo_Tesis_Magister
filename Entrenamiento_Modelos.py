import os
import pandas as pd
import numpy as np
import keras_tuner  as kt
from sklearn.preprocessing import MinMaxScaler
import random as python_random

os.chdir('C:/Codigo_Tesis_Magister/Referencia')
#from Funciones_2 import *                      # Importa toda funcion de Funciones_2.py
from Funciones_2 import PrePros                # Pre procesa los datos, normaliza y divide los datos
from Funciones_2 import split_sequences_clstm2 # Funcion split para ConvLSTM Intensidad + Desplazamiento
from Funciones_2 import split_sequences_clstm  # Funcion split para ConvLSTM solo Intensidad 
from Funciones_2 import split_sequences_lstm2  # Funcion split para LSTM Intensidad + Desplazamiento
from Funciones_2 import split_sequences_lstm   # Funcion split para LSTM solo Intensidad 
from Funciones_2 import split_sequences_ffnn2  # Funcion split para FFNN Intensidad + Desplazamiento
from Funciones_2 import split_sequences_ffnn   # Funcion split para FFNN solo Intensidad
from Funciones_2 import split_sequencesMC      # M-ConvLSTM
from Funciones_2 import build_fnn_1            # Constructor de modelo FFNN para Keras-tuner una variable
from Funciones_2 import build_fnn_2            # Constructor de modelo FFNN para Keras-tuner dos variables
from Funciones_2 import build_lstm_1           # Constructor de modelo LSTM para Keras-tuner  una variable
from Funciones_2 import build_lstm_2           # Constructor de modelo LSTM para Keras-tuner dos variables
from Funciones_2 import build_convlstm_1       # Constructor de modelo ConvLSTM para Keras-tuner una variable
from Funciones_2 import build_convlstm_2       # Constructor de modelo ConvLSTMpara Keras-tuner dos variables
from Funciones_2 import build_mconvlstm        # Constructor de modelo M-ConvLSTM para Keras-tuner
from Funciones_2 import ContModel              # Entrena modelo y guarda proceso de entrenamiento

scaler11 = MinMaxScaler()# Escala de 0 a 1
scaler12 = MinMaxScaler()

scaler21 = MinMaxScaler()
scaler22 = MinMaxScaler()

scaler31 = MinMaxScaler()
scaler32 = MinMaxScaler()

x1=np.log(np.loadtxt('INOB.txt').reshape(2922,20,20)) # Se cargan los datos
x2=np.loadtxt('DEOB.txt').reshape(2922,20,20)
    
for i in range(0,len(x1)): # transpongo todas las matrices para oreden de visualización
    x1[i]=np.transpose(x1[i])
    x2[i]=np.transpose(x2[i])

dates = pd.date_range('2009-12-31','2017-12-30',freq='D') # Se define intervalo de fechas
dates = dates.map(lambda t: t.strftime('%Y-%m-%d'))
dates=list(dates)

# Fechas de estudio
Idate1=['2009-12-31','2015-12-31']
Idate2=['2011-01-01','2016-12-31'] 
Idate3=['2012-01-01','2017-12-30']


# Prepara conjuntos de datos de intensidades y desplazamientos para c/a periodo

INA1,INE1=PrePros(x1, Idate1, dates, scaler11)
INA2,INE2=PrePros(x1, Idate2, dates, scaler21)
INA3,INE3=PrePros(x1, Idate3, dates, scaler31)

INA1.shape[0]+INE1.shape[0]

DEA1,DEE1=PrePros(x2, Idate1, dates, scaler12)
DEA2,DEE2=PrePros(x2, Idate2, dates, scaler22)
DEA3,DEE3=PrePros(x2, Idate3, dates, scaler32)

n_steps=20 # Pasos previos para hacer la prediccion
EpT=35     # Epocas para Tunear
BzT=20     # Batch size para Tunear
Dir='C:/Codigo_Tesis_Magister/Modelos' # Directorio
seed_value=3

# FFNN Intensidad + Desplazamiento
NomProy2='ffnn_2_p1'

x_train1 , y_train1=split_sequences_ffnn2(INA1,DEA1,n_steps)
x_test1  , y_test1 =split_sequences_ffnn2(INE1,DEE1,n_steps)


python_random.seed(seed_value)
tuner_ffnn_2_p1 = kt.Hyperband(build_fnn_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_ffnn_2_p1.search(x=x_train1 , y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
best_ffnn_2_p1=tuner_ffnn_2_p1.get_best_models(num_models=1)[0]
best_hps = tuner_ffnn_2_p1.get_best_hyperparameters(num_trials = 1)[0].values
best_ffnn_2_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_fnn_2,x_train1,y_train1,x_test1,y_test1,scaler11)


NomProy2='ffnn_2_p2'
x_train2 , y_train2=split_sequences_ffnn2(INA2,DEA2,n_steps)
x_test2  , y_test2 =split_sequences_ffnn2(INE2,DEE2,n_steps)

python_random.seed(seed_value)
tuner_ffnn_2_p2 = kt.Hyperband(build_fnn_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_ffnn_2_p2.search(x=x_train2 , y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
best_ffnn_2_p2=tuner_ffnn_2_p2.get_best_models(num_models=1)[0]
best_hps = tuner_ffnn_2_p2.get_best_hyperparameters(num_trials = 1)[0].values
best_ffnn_2_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_fnn_2,x_train2,y_train2,x_test2,y_test2,scaler21)

NomProy2='ffnn_2_p3'
x_train3 , y_train3=split_sequences_ffnn2(INA3,DEA3,n_steps)
x_test3  , y_test3 =split_sequences_ffnn2(INE3,DEE3,n_steps)

python_random.seed(seed_value)
tuner_ffnn_2_p3 = kt.Hyperband(build_fnn_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_ffnn_2_p3.search(x=x_train3 , y=y_train3,epochs=EpT,batch_size=BzT,verbose=0,validation_data=(x_test3, y_test3))
best_ffnn_2_p3=tuner_ffnn_2_p3.get_best_models(num_models=1)[0]
best_hps = tuner_ffnn_2_p3.get_best_hyperparameters(num_trials = 1)[0].values
best_ffnn_2_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_fnn_2,x_train3,y_train3,x_test3,y_test3,scaler31)

# FFNN solo Intensidad

NomProy2='ffnn_1_p1'

x_train1 , y_train1=split_sequences_ffnn(INA1,n_steps)
x_test1  , y_test1 =split_sequences_ffnn(INE1,n_steps)

python_random.seed(seed_value)
tuner_ffnn_1_p1 = kt.Hyperband(build_fnn_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_ffnn_1_p1.search(x=x_train1, y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
best_ffnn_1_p1=tuner_ffnn_1_p1.get_best_models(num_models=1)[0]
best_ffnn_1_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_fnn_1,x_train1,y_train1,x_test1,y_test1,scaler11)


NomProy2='ffnn_1_p2'

x_train2 , y_train2=split_sequences_ffnn(INA2,n_steps)
x_test2  , y_test2 =split_sequences_ffnn(INE2,n_steps)

python_random.seed(seed_value)
tuner_ffnn_1_p2 = kt.Hyperband(build_fnn_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_ffnn_1_p2.search(x=x_train2, y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
best_ffnn_1_p2=tuner_ffnn_1_p2.get_best_models(num_models=1)[0]
best_ffnn_1_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_fnn_1,x_train2,y_train2,x_test2,y_test2,scaler21)

NomProy2='ffnn_1_p3'
x_train3 , y_train3=split_sequences_ffnn(INA3,n_steps)
x_test3  , y_test3 =split_sequences_ffnn(INE3,n_steps)

python_random.seed(seed_value)
tuner_ffnn_1_p3 = kt.Hyperband(build_fnn_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_ffnn_1_p3.search(x=x_train3, y=y_train3,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test3, y_test3))
best_ffnn_1_p3=tuner_ffnn_1_p3.get_best_models(num_models=1)[0]
best_ffnn_1_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_fnn_1,x_train3,y_train3,x_test3,y_test3,scaler31)

# LSTM Intensidad + Desplazamiento

NomProy2='lstm_2_p1'
x_train1 , y_train1=split_sequences_lstm2(INA1,DEA1,n_steps)
x_test1  , y_test1 =split_sequences_lstm2(INE1,DEE1,n_steps)

python_random.seed(seed_value)
tuner_lstm_2_p1 = kt.Hyperband(build_lstm_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_lstm_2_p1.search(x=x_train1, y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
best_lstm_2_p1=tuner_lstm_2_p1.get_best_models(num_models=1)[0]
best_lstm_2_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_lstm_2,x_train1,y_train1,x_test1,y_test1,scaler11)


NomProy2='lstm_2_p2'
x_train2 , y_train2=split_sequences_lstm2(INA2,DEA2,n_steps)
x_test2  , y_test2 =split_sequences_lstm2(INE2,DEE2,n_steps)

python_random.seed(seed_value)
tuner_lstm_2_p2 = kt.Hyperband(build_lstm_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_lstm_2_p2.search(x=x_train2, y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
best_lstm_2_p2=tuner_lstm_2_p2.get_best_models(num_models=1)[0]
best_lstm_2_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_lstm_2,x_train2,y_train2,x_test2,y_test2,scaler21)

NomProy2='lstm_2_p3'
x_train3 , y_train3=split_sequences_lstm2(INA3,DEA3,n_steps)
x_test3  , y_test3 =split_sequences_lstm2(INE3,DEE3,n_steps)

python_random.seed(seed_value)
tuner_lstm_2_p3 = kt.Hyperband(build_lstm_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_lstm_2_p3.search(x=x_train3, y=y_train3,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test3, y_test3))
best_lstm_2_p3=tuner_lstm_2_p3.get_best_models(num_models=1)[0]
best_lstm_2_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_lstm_2,x_train3,y_train3,x_test3,y_test3,scaler31)

# LSTM solo Intensidad
 
NomProy2='lstm_1_p1'
x_train1 , y_train1=split_sequences_lstm(INA1,n_steps)
x_test1  , y_test1 =split_sequences_lstm(INE1,n_steps)

python_random.seed(seed_value)
tuner_lstm_1_p1 = kt.Hyperband(build_lstm_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_lstm_1_p1.search(x=x_train1, y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
best_lstm_1_p1=tuner_lstm_1_p1.get_best_models(num_models=1)[0]
best_lstm_1_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_lstm_1,x_train1,y_train1,x_test1,y_test1,scaler11)

NomProy2='lstm_1_p2'
x_train2 , y_train2=split_sequences_lstm(INA2,n_steps)
x_test2  , y_test2 =split_sequences_lstm(INE2,n_steps)

python_random.seed(seed_value)
tuner_lstm_1_p2 = kt.Hyperband(build_lstm_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name='lstm_1_p2')

tuner_lstm_1_p2.search(x=x_train2, y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
best_lstm_1_p2=tuner_lstm_1_p2.get_best_models(num_models=1)[0]
best_lstm_1_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_lstm_1,x_train2,y_train2,x_test2,y_test2,scaler21)

NomProy2='lstm_1_p3'
x_train3 , y_train3=split_sequences_lstm(INA3,n_steps)
x_test3  , y_test3 =split_sequences_lstm(INE3,n_steps)

python_random.seed(seed_value)
tuner_lstm_1_p3 = kt.Hyperband(build_lstm_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_lstm_1_p3.search(x=x_train3, y=y_train3,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test3, y_test3))
best_lstm_1_p3=tuner_lstm_1_p3.get_best_models(num_models=1)[0]
best_lstm_1_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_lstm_1,x_train3,y_train3,x_test3,y_test3,scaler31)


# ConvLSTM Intensidad + Desplazamiento
NomProy2='convlstm_2_p1'
x_train1 , y_train1=split_sequences_clstm2(INA1,DEA1,n_steps)
x_test1  , y_test1 =split_sequences_clstm2(INE1,DEE1,n_steps)

python_random.seed(seed_value)
tuner_convlstm_2_p1 = kt.Hyperband(build_convlstm_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_convlstm_2_p1.search(x=x_train1, y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
tuner_convlstm_2_p1=tuner_convlstm_2_p1.get_best_models(num_models=1)[0]
tuner_convlstm_2_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_convlstm_2,x_train1,y_train1,x_test1,y_test1,scaler11)

NomProy2='convlstm_2_p2'
x_train2 , y_train2=split_sequences_clstm2(INA2,DEA2,n_steps)
x_test2  , y_test2 =split_sequences_clstm2(INE2,DEE2,n_steps)

python_random.seed(seed_value)
tuner_convlstm_2_p2 = kt.Hyperband(build_convlstm_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_convlstm_2_p2.search(x=x_train2, y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
tuner_convlstm_2_p2=tuner_convlstm_2_p2.get_best_models(num_models=1)[0]
tuner_convlstm_2_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_convlstm_2,x_train2,y_train2,x_test2,y_test2,scaler21)


NomProy2='convlstm_2_p3'
x_train3 , y_train3=split_sequences_clstm2(INA3,DEA3,n_steps)
x_test3  , y_test3 =split_sequences_clstm2(INE3,DEE3,n_steps)

python_random.seed(seed_value)
tuner_convlstm_2_p3 = kt.Hyperband(build_convlstm_2,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_convlstm_2_p3.search(x=x_train3, y=y_train3,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test3, y_test3))
tuner_convlstm_2_p3=tuner_convlstm_2_p3.get_best_models(num_models=1)[0]
tuner_convlstm_2_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_convlstm_2,x_train3,y_train3,x_test3,y_test3,scaler31)

# ConvLSTM solo Intensidad 

NomProy2='convlstm_1_p1'
x_train1 , y_train1=split_sequences_clstm(INA1,n_steps)
x_test1  , y_test1 =split_sequences_clstm(INE1,n_steps)

python_random.seed(seed_value)
tuner_convlstm_1_p1 = kt.Hyperband(build_convlstm_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_convlstm_1_p1.search(x=x_train1, y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
tuner_convlstm_1_p1=tuner_convlstm_1_p1.get_best_models(num_models=1)[0]
tuner_convlstm_1_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_convlstm_1,x_train1,y_train1,x_test1,y_test1,scaler11)

NomProy2='convlstm_1_p2'
x_train2 , y_train2=split_sequences_clstm(INA2,n_steps)
x_test2  , y_test2 =split_sequences_clstm(INE2,n_steps)

python_random.seed(seed_value)
tuner_convlstm_1_p2 = kt.Hyperband(build_convlstm_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_convlstm_1_p2.search(x=x_train2, y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
tuner_convlstm_1_p2=tuner_convlstm_1_p2.get_best_models(num_models=1)[0]
tuner_convlstm_1_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_convlstm_1,x_train2,y_train2,x_test2,y_test2,scaler21)

NomProy2='convlstm_1_p3'
x_train3 , y_train3=split_sequences_clstm(INA3,n_steps)
x_test3  , y_test3 =split_sequences_clstm(INE3,n_steps)

python_random.seed(seed_value)
tuner_convlstm_1_p3 = kt.Hyperband(build_convlstm_1,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_convlstm_1_p3.search(x=x_train3, y=y_train3,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test3, y_test3))
tuner_convlstm_1_p3=tuner_convlstm_1_p3.get_best_models(num_models=1)[0]
tuner_convlstm_1_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_convlstm_1,x_train3,y_train3,x_test3,y_test3,scaler31)

# M-ConvLSTM 

NomProy2='mconvlstm_p1'
x_train1 , y_train1=split_sequencesMC(INA1,DEA1,n_steps)
x_test1  , y_test1 =split_sequencesMC(INE1,DEE1,n_steps)

python_random.seed(seed_value)
tuner_mconvlstm_p1 = kt.Hyperband(build_mconvlstm,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_mconvlstm_p1.search(x=x_train1, y=y_train1,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test1, y_test1))
tuner_mconvlstm_p1=tuner_mconvlstm_p1.get_best_models(num_models=1)[0]
tuner_mconvlstm_p1.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_mconvlstm,x_train1,y_train1,x_test1,y_test1,scaler11)

NomProy2='mconvlstm_p2'
x_train2 , y_train2=split_sequencesMC(INA2,DEA2,n_steps)
x_test2  , y_test2 =split_sequencesMC(INE2,DEE2,n_steps)

python_random.seed(seed_value)
tuner_mconvlstm_p2 = kt.Hyperband(build_mconvlstm,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_mconvlstm_p2.search(x=x_train2, y=y_train2,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test2, y_test2))
tuner_mconvlstm_p2=tuner_mconvlstm_p2.get_best_models(num_models=1)[0]
tuner_mconvlstm_p2.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_mconvlstm,x_train2,y_train2,x_test2,y_test2,scaler21)

NomProy2='mconvlstm_p3'
x_train3 , y_train3=split_sequencesMC(INA3,DEA3,n_steps)
x_test3  , y_test3 =split_sequencesMC(INE3,DEE3,n_steps)

python_random.seed(seed_value)

tuner_mconvlstm_p3 = kt.Hyperband(build_mconvlstm,
                     objective = 'val_mse', 
                     max_epochs = 40,
                     factor = 3,
                     hyperband_iterations=20,
                     directory = Dir,
                     project_name=NomProy2)

tuner_mconvlstm_p3.search(x=x_train3, y=y_train3,epochs=EpT,batch_size=BzT,verbose=2,validation_data=(x_test3, y_test3))
tuner_mconvlstm_p3=tuner_mconvlstm_p3.get_best_models(num_models=1)[0]
tuner_mconvlstm_p3.save('best_'+NomProy2+'.h5', include_optimizer=False)
ContModel(NomProy2,Dir,build_mconvlstm,x_train3,y_train3,x_test3,y_test3,scaler31)

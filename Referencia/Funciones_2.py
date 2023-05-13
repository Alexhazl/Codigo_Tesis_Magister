import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.initializers import GlorotNormal
from keras.layers import Concatenate, Input
from keras.models import Model
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import kerastuner  as kt
import pandas as pd

seed_value=3

def rsq(yobs,ypred):
  ym=np.mean(yobs)
  num=np.sum((yobs-ypred)**2)
  den=np.sum((yobs-ym)**2)
  r2=1-(num/den)
  return r2 

def PrePros(df,Idate,dates,scaler):
    df=df.reshape(df.shape[0],20*20)
    idate=[dates.index(Idate[0]),dates.index(Idate[1])+1]
    df=df[idate[0]:idate[1]]
    df= scaler.fit_transform(df) # Normalización
    df=df.reshape(df.shape[0],20,20)
    train_size = int(len(df) * 0.83)# Dimensiones para el entrenamiento y test
    df1 = df[0:train_size,:]
    df2 = df[train_size:len(df),:]
    return df1,df2

def split_sequences2(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def split_sequencesMC(sequences1,sequences2, n_steps):
    X=list()
    x1 , y = split_sequences2(sequences1,n_steps)
    x2, NC = split_sequences2(sequences2,n_steps)
    
    x1=x1.reshape(len(x1),n_steps,20,20,1)
    x2=x2.reshape(len(x2),n_steps,20,20,1)
    
    y=y.reshape(len(y),20,20,1)
    X=[x1,x2]
    
    return X,y 

def split_sequences_ffnn2 (sequences1,sequences2, n_steps):
    sequences1=sequences1.reshape(sequences1.shape[0],20*20)
    sequences2=sequences2.reshape(sequences2.shape[0],20*20)
    
    sequences3=np.stack((sequences1,sequences2),axis=2)
    sequences3=sequences3.reshape(len(sequences3),sequences3.shape[1]*sequences3.shape[2])

    X, y = list(), list()
    for i in range(len(sequences1)):
        end_ix = i + n_steps
        if end_ix > len(sequences1)-1:
            break
        seq_x, seq_y = sequences3[end_ix-1, :], sequences1[end_ix ,:]
        X.append(seq_x)
        y.append(seq_y)
    y=np.array(y)
    y=y.reshape(y.shape[0],y.shape[1])
    return np.array(X), y

def split_sequences_ffnn (sequences1, n_steps):
    sequences1=sequences1.reshape(sequences1.shape[0],20*20)
    
    X, y = list(), list()
    for i in range(len(sequences1)):
        end_ix = i + n_steps
        if end_ix > len(sequences1)-1:
            break
        seq_x, seq_y = sequences1[end_ix-1, :], sequences1[end_ix ,:]
        X.append(seq_x)
        y.append(seq_y)
    y=np.array(y)
    return np.array(X), y

def split_sequences_lstm2 (sequences1,sequences2, n_steps):
    sequences1=sequences1.reshape(sequences1.shape[0],20*20)
    sequences2=sequences2.reshape(sequences2.shape[0],20*20)
    
    sequences3=np.stack((sequences1,sequences2),axis=2)
    sequences3=sequences3.reshape(len(sequences3),sequences3.shape[1]*sequences3.shape[2])
    
    X, y = list(), list()
    for i in range(len(sequences1)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences1)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences3[i:end_ix, :], sequences1[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    
    y=np.array(y)
    return np.array(X), y

def split_sequences_clstm2 (sequences1,sequences2, n_steps):
    #n=1
    sequences3=np.stack((sequences1,sequences2),axis=3)
    X, y = list(), list()
    for i in range(len(sequences1)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences1)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences3[i:end_ix, :], sequences1[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    
    y=np.array(y)
    y=y.reshape(y.shape[0],y.shape[1],y.shape[2],1)
    return np.array(X), y

def split_sequences_clstm (sequences1, n_steps):
    X, y = list(), list()
    for i in range(len(sequences1)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences1)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences1[i:end_ix, :], sequences1[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    
    y=np.array(y)
    X=np.array(X)
    y=y.reshape(y.shape[0],y.shape[1],y.shape[2],1)
    X=X.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],1)

    return X, y

def split_sequences_lstm (sequences1, n_steps):
    sequences1=sequences1.reshape(len(sequences1),20*20)
    X, y = list(), list()
    for i in range(len(sequences1)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences1)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences1[i:end_ix, :], sequences1[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    
    y=np.array(y)
    X=np.array(X)

    return X, y

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - (SS_res/SS_tot))

def M_LSTM(Neu=[40,40],DO=[0.2,0.2],n_steps=20, n_features=800,n_out=400,Seed=3):
    initializer = GlorotNormal(seed=Seed)
    model = Sequential()
    model.add(LSTM(Neu[0], activation='tanh',return_sequences=True,kernel_initializer=initializer ,input_shape=(n_steps, n_features)))
    model.add(Dropout(DO[0]))
    model.add(LSTM(Neu[1],activation='tanh',return_sequences=True,kernel_initializer=initializer))  
    model.add(Dropout(DO[1]))
    model.add(Dense(n_out,activation='linear',kernel_initializer=initializer))    
    return model

def M_FFNN(Neu=[500,200,200],DO=[0.2,0.2], n_features=800,n_out=400,Seed=3):
    initializer = GlorotNormal(seed=Seed)
    model = Sequential()
    model.add(Dense(Neu[0], input_dim=n_features,kernel_initializer=initializer,activation='linear'))
    model.add(Dropout(DO[0]))
    model.add(Dense(Neu[1],kernel_initializer=initializer,activation='linear'))
    model.add(Dropout(DO[1]))
    model.add(Dense(Neu[2],activation='linear',kernel_initializer=initializer))
    model.add(Dense(n_out,kernel_initializer=initializer))
    
    return model

def MCL_body_branch(input_flow,FLt1=[5,3],DO1=[0.5,0.5],KS1=[(4,4),(8,8)],Seed=3):
    initializer = GlorotNormal(seed=Seed)
    x=ConvLSTM2D(filters=FLt1[0], kernel_size=KS1[0],kernel_initializer=initializer,padding='same', return_sequences=True)(input_flow)
    x=Dropout(DO1[0])(x)
    x=BatchNormalization()(x)
    x=ConvLSTM2D(filters=FLt1[1], kernel_size=KS1[1],padding='same',return_sequences=True,kernel_initializer=initializer)(x)
    x=Dropout(DO1[1])(x)
    x=BatchNormalization()(x)
    x=Model(inputs=input_flow, outputs=x)

    return x

def MCL( input_shape = (20, 20, 20, 1),FLt=[5,3],DO=[0.5,0.5],KS=[(4,4),(8,8),(4,4)],Seed=3):
    initializer = GlorotNormal(seed=Seed)
    input_flow1 = Input(shape=input_shape)
    input_flow2 = Input(shape=input_shape)
    a=MCL_body_branch(input_flow=input_flow1,FLt1=FLt,DO1=DO,KS1=KS[0:2])
    b=MCL_body_branch(input_flow=input_flow2,FLt1=FLt,DO1=DO,KS1=KS[0:2])
    combined = Concatenate(axis=4)([a.output, b.output])
    z=ConvLSTM2D(filters=1, kernel_size=KS[2],activation='linear',padding='same',kernel_initializer=initializer)(combined)
    model = Model(inputs=[a.input, b.input], outputs=z)
   
    return model

def M_ConvLSTM(FLt=[3,4],KS=[(10,10),(6,6),(4,4)],DO=[0.5,0.5],Seed=3,n_steps=20,n_features=2):
    img_height=20
    img_width=20
    initializer = GlorotNormal(seed=Seed)

    model = Sequential()
    model.add(ConvLSTM2D(filters=FLt[0], kernel_size=KS[0],kernel_initializer=initializer,
                input_shape = (n_steps, img_height, img_width,n_features),padding='same', return_sequences=True))#20
    model.add(Dropout(DO[0]))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=FLt[1], kernel_size=KS[1],padding='same',return_sequences=True,kernel_initializer=initializer))#10
    model.add(Dropout(DO[1]))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=1, kernel_size=KS[2],activation='linear',padding='same',kernel_initializer=initializer))

    return model

def build_fnn_2(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 50, max_value = 1000, step = 50)
    hp_units2 =hp.Int('units_2',  min_value = 50, max_value = 1000, step = 50)
    hp_units3 =hp.Int('units_3',  min_value = 50, max_value = 1000, step = 50)

    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=M_FFNN(Neu=[hp_units1,hp_units2,hp_units3],DO=[hp_dp_rate1,hp_dp_rate2], n_features=800,n_out=400,Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def build_fnn_1(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 50, max_value = 1000, step = 50)
    hp_units2 =hp.Int('units_2',  min_value = 50, max_value = 1000, step = 50)
    hp_units3 =hp.Int('units_3',  min_value = 50, max_value = 1000, step = 50)

    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=M_FFNN(Neu=[hp_units1,hp_units2,hp_units3],DO=[hp_dp_rate1,hp_dp_rate2], n_features=400,n_out=400,Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def build_lstm_2(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 50, max_value = 1000, step = 50)
    hp_units2 =hp.Int('units_2',  min_value = 50, max_value = 1000, step = 50)

    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=M_LSTM(Neu=[hp_units1,hp_units2],DO=[hp_dp_rate1,hp_dp_rate2], n_features=800,n_out=400,Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def build_lstm_1(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 50, max_value = 1000, step = 50)
    hp_units2 =hp.Int('units_2',  min_value = 50, max_value = 1000, step = 50)

    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=M_LSTM(Neu=[hp_units1,hp_units2],DO=[hp_dp_rate1,hp_dp_rate2], n_features=400,n_out=400,Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def build_convlstm_1(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 1, max_value = 50, step = 10)
    hp_units2 =hp.Int('units_2',  min_value = 1, max_value = 50, step = 10)

    hp_KS1 =hp.Int('KS_1',  min_value = 1, max_value = 10, step = 1)
    hp_KS2 =hp.Int('KS_2',  min_value = 1, max_value = 10, step = 1)
    hp_KS3 =hp.Int('KS_3',  min_value = 1, max_value = 10, step = 1)
    
    KS1=(hp_KS1,hp_KS1)
    KS2=(hp_KS2,hp_KS2)
    KS3=(hp_KS3,hp_KS3)
    
    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=M_ConvLSTM(FLt=[hp_units1,hp_units2],KS=[KS1,KS2,KS3],DO=[hp_dp_rate1,hp_dp_rate2], n_features=1,Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def build_convlstm_2(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 1, max_value = 50, step = 10)
    hp_units2 =hp.Int('units_2',  min_value = 1, max_value = 50, step = 10)

    hp_KS1 =hp.Int('KS_1',  min_value = 1, max_value = 10, step = 1)
    hp_KS2 =hp.Int('KS_2',  min_value = 1, max_value = 10, step = 1)
    hp_KS3 =hp.Int('KS_3',  min_value = 1, max_value = 10, step = 1)
    
    KS1=(hp_KS1,hp_KS1)
    KS2=(hp_KS2,hp_KS2)
    KS3=(hp_KS3,hp_KS3)
    
    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=M_ConvLSTM(FLt=[hp_units1,hp_units2],KS=[KS1,KS2,KS3],DO=[hp_dp_rate1,hp_dp_rate2], n_features=2,Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def build_mconvlstm(hp):
    seed_value=3
    tf.random.set_seed(seed_value)
    hp_units1 =hp.Int('units_1',  min_value = 1, max_value = 50, step = 10)
    hp_units2 =hp.Int('units_2',  min_value = 1, max_value = 50, step = 10)

    hp_KS1 =hp.Int('KS_1',  min_value = 1, max_value = 10, step = 1)
    hp_KS2 =hp.Int('KS_2',  min_value = 1, max_value = 10, step = 1)
    hp_KS3 =hp.Int('KS_3',  min_value = 1, max_value = 10, step = 1)
    
    KS1=(hp_KS1,hp_KS1)
    KS2=(hp_KS2,hp_KS2)
    KS3=(hp_KS3,hp_KS3)
    
    hp_dp_rate1 = hp.Choice('dp_rate1', values = [0.2, 0.25, 0.5, 0.8]) 
    hp_dp_rate2 = hp.Choice('dp_rate2', values = [0.2, 0.25, 0.5, 0.8]) 

    hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.01, 0.1]) 

    model=MCL(FLt=[hp_units1,hp_units2],KS=[KS1,KS2,KS3],DO=[hp_dp_rate1,hp_dp_rate2],Seed=3)
    optimizer1 = Adam(lr=hp_learning_rate)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mse','mae','mean_absolute_percentage_error',coeff_determination])
    
    return model

def ContModel(NomProy,direc,Bulid,Xa,Ya,Xe,Ye,scal):
    dependencies = {'coeff_determination': coeff_determination}
    model = load_model('best_'+NomProy+'.h5',custom_objects=dependencies)
    tuner = kt.Hyperband(Bulid,
                         objective = 'val_mse', 
                         max_epochs = 40,
                         factor = 3,
                         hyperband_iterations=20,
                         directory = direc,
                         project_name=NomProy)
    tuner.reload()
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0].values
    LRt=best_hps['learning_rate']

    optimizer1 = Adam(lr=LRt)
    model.compile(optimizer=optimizer1, loss='mse',metrics=['mean_absolute_percentage_error','mae',coeff_determination])# Agregar el coefficiente de determinacion 
    tf.random.set_seed(3)
    model.fit(Xa,Ya, epochs=220, batch_size=20 ,verbose=2,validation_data=(Xe, Ye))

    trainPredict = model.predict(Xa)
    testPredict =  model.predict(Xe)

    a1='%.3f' % mean_absolute_error(Ya,trainPredict)
    b1='%.3f' % mean_squared_error(Ya,trainPredict)
    c1='%.3f' % rsq(Ya,trainPredict)
    
    a2='%.3f' % mean_absolute_error(Ye,testPredict)
    b2='%.3f' % mean_squared_error(Ye,testPredict)
    c2='%.3f' % rsq(Ye,testPredict)
    
    d = {'MAE': [a1, a2], 'MSE': [b1, b2],'R2':[c1,c2]}
    df = pd.DataFrame(data=d,index=["Train", "Test"])
    df.to_csv(r'Metrics_'+NomProy+'.txt', sep='\t')
    
    if(len(testPredict.shape)!=2):
        trainPredict=trainPredict.reshape(trainPredict.shape[0],20*20)
        testPredict=testPredict.reshape(testPredict.shape[0],20*20)
        Ya=Ya.reshape(Ya.shape[0],20*20)
        Ye=Ye.reshape(Ye.shape[0],20*20)
    
    trainPredict=np.exp(scal.inverse_transform(trainPredict))
    testPredict =np.exp(scal.inverse_transform(testPredict))
    
    Ya=np.exp(scal.inverse_transform(Ya))
    Ye=np.exp(scal.inverse_transform(Ye))
    
    np.savetxt('TrainP_'+NomProy+'.txt', trainPredict, fmt='%.5f')   
    np.savetxt('TestP_'+NomProy+'.txt', testPredict, fmt='%.5f')   
    
    np.savetxt('ObsTrain.txt', Ya, fmt='%.5f')   
    np.savetxt('ObsTest.txt', Ye, fmt='%.5f')   
    
    
    
    model.save('Final_'+NomProy+'.h5', include_optimizer=False)
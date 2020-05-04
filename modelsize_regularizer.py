# import packages
import keras
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from morph_net.network_regularizers import flop_regularizer
from morph_net.network_regularizers import model_size_regularizer
from morph_net.tools import structure_exporter
import keras.backend as K

import logging
tf.get_logger().setLevel(logging.ERROR)


   
class modelsize_regularizer:
    def __init__(self,ops,regularizer_strength=1e-3,**kwargs):
        self._network_regularizer = model_size_regularizer.GroupLassoModelSizeRegularizer(ops, **kwargs)
        self._regularization_strength = regularizer_strength
        self._regularizer_loss = (self._network_regularizer.get_regularization_term() * self._regularization_strength)    
        self._sess = K.get_session()
        
    def loss(self,*args):
        with self._sess.as_default():
            return keras.losses.mean_squared_error(*args) + self._regularizer_loss.eval()
        
    def flops(self,*args):
        return self._network_regularizer.get_cost()
        
    def regularizer_loss(self,*args):
        return self._regularizer_loss

def import_dataset():
    x_dataset = pd.DataFrame(columns=['x','xdot','theta','theta_dot'])
    y_dataset = pd.DataFrame(columns=['u'])
    
    temp1 = pd.DataFrame(columns=['x','xdot','theta','theta_dot'])
    temp2 = pd.DataFrame(columns=['u'])
    
    se = 'states\states_0.csv'
    ce = 'control\control_0.csv'
    for i in range(99):
        z=0
        temp=''
        while(se[14+z].isdigit()):
            temp+=se[14+z]
            z+=1
        temp = str(int(temp)+1)
        se = se[:14]+temp+se[14+z:]
        ce = ce[:16]+temp+ce[16+z:]
        temp1 = pd.read_csv(se)
        temp2 = pd.read_csv(ce)
        x_dataset = x_dataset.append(temp1.iloc[:5800,1:])
        y_dataset = y_dataset.append(temp2.iloc[:5800,1:])    
        
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
    
def save_model(model, y_pred, x_test, fname):
    model.save(fname)
    np.save('Ypred', y_pred)
    np.save('Xtest', x_test)

def flop_memory_estimate(model):
    def count_dense_params_flops(dense_layer, verbose=1):
        # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
        out_shape = dense_layer.output.shape.as_list()
        n_cells_total = np.prod(out_shape[1:-1])
    
        n_dense_params_total = dense_layer.count_params()
        
        dense_flops = 2 * n_dense_params_total
    
        return n_dense_params_total, dense_flops

    total_flops = 0
    total_params = 0
    for i in range(len(model.layers)):
        n_dense_params_total, dense_flops = count_dense_params_flops(model.layers[i])
        total_flops+=dense_flops
        total_params+=n_dense_params_total
       
    shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))
    memory = shapes_count * 4
    print("Memory in bytes: ",memory," Total_Flops: ",total_flops)
 
def neural_network():
    x_train, x_test, y_train, y_test = import_dataset()
    model = Sequential()
    model.add(Dense(units=32,activation='relu',input_shape=(x_train.shape[1],)))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mean_absolute_error'])
    history1 =  model.fit(x_train.values, y_train.values, batch_size=200, epochs=10, verbose=0, validation_data=(x_test,y_test))
    y_pred1 = model.predict(x_test)
    print("FLOP and Memory Estimate for Seed Network:")
    flop_memory_estimate(model)
    
    morphnet = modelsize_regularizer([model.output.op], threshold=1e-3)
    model.compile(optimizer='adam',loss=morphnet.loss, metrics=['mean_absolute_error',morphnet.flops, morphnet.regularizer_loss])
    history2 = model.fit(x_train.values, y_train.values, batch_size=200, epochs=10, verbose=0, validation_data=(x_test,y_test))
    
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mean_absolute_error'])
    history3 =  model.fit(x_train.values, y_train.values, batch_size=200, epochs=10, verbose=0, validation_data=(x_test,y_test))
    y_pred2 = model.predict(x_test)
    
    save_model(model, y_pred2, x_test, "modelsize_model.h5")
    
    plt.plot(history1.history['val_loss'])
    plt.plot(history3.history['val_loss'])
    plt.title('Learning Curves')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['seed network', 'ModelSize optimized network'], loc='upper center')
    plt.figure()
    
    plt.plot(y_test, y_pred1)
    plt.plot(y_test, y_pred2)
    plt.title('Prediction Comparisons')
    plt.xlabel('inputs')
    plt.ylabel('targets')
    plt.legend(['seed network', 'ModelSize Optimized Network' ], loc='lower right')
    plt.show()
    
    print("FLOP and Memory Estimate for Optimized Network:")
    flop_memory_estimate(model)
    

    
neural_network()    
    
    
    
    
    
    
    
    
    
    
    
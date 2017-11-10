# @author: gmishne
import numpy as np
import keras.backend as K
import tensorflow as tf

from keras.utils import np_utils
from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers

#  pre-training autoencoders functions

#  activity regularization using KL divergence
def kl_divergence(p, p_hat):
    return K.mean((p * K.log(p / p_hat)) + ((1-p) * K.log((1-p) / (1-p_hat))))

class SparseActivityRegularizer(regularizers.Regularizer):
    sparsityBeta = None
    
    def __init__(self, l1=0., l2=0., p=0.1, sparsityBeta=1):
        self.p = p
        self.sparsityBeta = sparsityBeta
    
    def __call__(self, x):
        loss = 0.
        #p_hat is the average activation of the units in the hidden layer.
        p_hat = K.mean(x,axis=0)
        
        loss += self.sparsityBeta *(kl_divergence(self.p, p_hat))
        return loss
    
    def get_config(self):
        return {"name": self.__class__.__name__,
            "p": self.l1}

class Autoencoder:

    def __init__(self,input_size, hidden_size,hidden_activation='sigmoid', output_activation='sigmoid',reg_par=1e-4,beta=0.01,p=0.1):
        
        input_data = Input(shape=(input_size,))
        encoded = Dense(hidden_size, activation=hidden_activation, kernel_initializer='glorot_uniform', activity_regularizer=SparseActivityRegularizer(sparsityBeta=beta,p=p),
            kernel_regularizer=regularizers.l2(reg_par))(input_data)

        decoded = Dense(input_size, activation=output_activation, kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(reg_par))(encoded)
            
        self.autoencoder = Model(input_data,decoded)
        self.encoder = Model(input_data,encoded)

    def compile(self,optimizer='adam'):
        self.autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    def train(self,data,batch_size,n_epochs=400):
        self.autoencoder.fit(data, data,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=0)

    def encode(self,data):
        return self.encoder.predict(data)

    def predict(self,data):
        return self.autoencoder.predict(data)

    def get_weights(self):
        return self.autoencoder.layers[1].get_weights()

# single layer linear regression
def pretrain_regression(data, target, input_size, hidden_size, batch_size, reg_par=1e-4,
                        n_epochs=400):
    input_data = Input(shape=(input_size,))
    encoded = Dense(hidden_size, activation='linear',kernel_initializer='glorot_uniform',
                    activity_regularizer=regularizers.l2(reg_par),
                    kernel_regularizer=regularizers.l2(reg_par))(input_data)
    
    reg = Model(input_data, encoded)
    reg.compile(optimizer='adam', loss='mean_squared_error')
    
    reg.fit(data, target,
            epochs=n_epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0)
                    
    return reg

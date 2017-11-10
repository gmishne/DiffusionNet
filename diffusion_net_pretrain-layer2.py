# @author: gmishne
# coding: utf-8
import random

import numpy as np
import keras.backend as K
import tensorflow as tf

import Diffusion as df
import os.path
from autoencoder import *

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.utils import np_utils
from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers


# ##  generate data #######################
# 
random.seed(101017)

n = 3000
t = 2.01*np.pi*np.random.uniform(0,1,n)
x = np.cos(t)
y = np.sin(2*t)
z = np.sin(3*t)
orig_data = np.vstack([x,y,z])
orig_data = np.transpose(orig_data)

noise_std = 0.05
data = orig_data + noise_std*np.random.randn(orig_data.shape[0],orig_data.shape[1])

n_train = 2000
n_test = n - n_train

S1_train   = data[:n_train,:]
S1_test    = data[n_train+1:,:]
clean_test = orig_data[n_train+1:,:]
t_train    = t[:n_train]
t_test     = t[n_train+1:]
input_size = S1_train.shape[1]
batch_size = S1_train.shape[0]

sort_inds = np.argsort(t_train)

embedding_size = 2
k = 16
K_mat = df.ComputeLBAffinity(S1_train,k,sig=0.1)   # Laplace-Beltrami affinity: D^-1 * K * D^-1
P     = df.makeRowStoch(K_mat)                     # markov matrix 
E1,v1 = df.Diffusion(K_mat, nEigenVals=embedding_size+1)  # eigenvalues and eigenvectors
S1_embedding = np.matmul(E1, np.diag(v1)) # diffusion maps

fig, (a1)  = plt.subplots(1,1)
a1.scatter(E1[:,0], E1[:,1], c=t_train, cmap='gist_ncar')
plt.title('diffusion embedding of train')
a1.set_aspect('equal')
plt.show(block=False)
plt.savefig('DM2layer_embedding_.png', bbox_inches='tight',transparent = True)

# # Diffusion Net

P  = tf.cast(tf.constant(P),tf.float32)
E1 = tf.cast(tf.constant(E1),tf.float32)
v1 = tf.cast(tf.constant(v1),tf.float32)


# ### config net #######################
# 

N_EPOCHS = 1000

# number of units in encoder and decoder
encoder_layer_sizes = [20, 20, embedding_size]
decoder_layer_sizes = [20, 20, input_size]

# ## Pre-train encoder
print "Start pre-train Encoder"
autoencoder1 = Autoencoder(input_size=input_size, hidden_size=encoder_layer_sizes[0], 
                            reg_par=1e-7,output_activation='linear')

autoencoder1.compile(optimizer='adam')
autoencoder1.train(S1_train,batch_size=n_train,n_epochs=N_EPOCHS)
output1 = autoencoder1.predict(S1_train)

encoder1_train = autoencoder1.encode(S1_train)

fig = plt.figure(figsize=(10,4))
a1 = fig.add_subplot(121,projection='3d')
a1.scatter(S1_train[:,0], S1_train[:,1], S1_train[:,2], c=t_train, cmap='gist_ncar')
plt.title('training data')
a2 = fig.add_subplot(122,projection='3d')
a2.scatter(output1[:,0], output1[:,1], output1[:,2], c=t_train, cmap='gist_ncar')
plt.title('1st layer pre-trained AE output')
plt.show(block=False)

autoencoder2 = Autoencoder(input_size=encoder_layer_sizes[0], hidden_size=encoder_layer_sizes[1], reg_par=1e-7)

autoencoder2.compile(optimizer='adam')
autoencoder2.train(encoder1_train,batch_size=n_train,n_epochs=N_EPOCHS)

encoder2_train = autoencoder2.encode(encoder1_train)

encoder3 = pretrain_regression(encoder2_train, S1_embedding, encoder_layer_sizes[1], encoder_layer_sizes[2],
                               batch_size, reg_par=1e-4, n_epochs=N_EPOCHS) 

encoder3_train = encoder3.predict(encoder2_train)

fig, (a2)  = plt.subplots(1,1)
a2.scatter(encoder3_train[:,0], encoder3_train[:,1], c=t_train, cmap='gist_ncar')
plt.axis('equal')
plt.title('diffusion net encoder output')
plt.show(block=False)


# ## Pre-train decoder
print "Start pre-train Deccoder"
de_autoencoder1 = Autoencoder(input_size=embedding_size, hidden_size=decoder_layer_sizes[0], 
                            reg_par=1e-4,output_activation='linear')

de_autoencoder1.compile(optimizer='adam')
de_autoencoder1.train(S1_embedding,batch_size=n_train,n_epochs=N_EPOCHS)
de_output1 = de_autoencoder1.predict(S1_embedding)

de_encoder1_train = de_autoencoder1.encode(S1_embedding)

fig = plt.figure(figsize=(10,4))
a1 = fig.add_subplot(121)
a1.scatter(S1_embedding[:,0], S1_embedding[:,1], c=t_train, cmap='gist_ncar')
plt.title('diffusion embedding of train')
a1.set_aspect('equal')
a2 = fig.add_subplot(122)
a2.scatter(de_output1[:,0], de_output1[:,1], c=t_train, cmap='gist_ncar')
a2.set_aspect('equal')
plt.title('1st layer pre-trained AE output')
plt.show(block=False)

de_autoencoder2 = Autoencoder(input_size=decoder_layer_sizes[0], hidden_size=decoder_layer_sizes[1],
                            reg_par=1e-4)

de_autoencoder2.compile(optimizer='adam')
de_autoencoder2.train(de_encoder1_train,batch_size=n_train,n_epochs=N_EPOCHS)
de_encoder2_train = de_autoencoder2.encode(de_encoder1_train)

de_encoder3 = pretrain_regression(de_encoder2_train, S1_train, decoder_layer_sizes[1], decoder_layer_sizes[2],
                               batch_size, reg_par=1e-4, n_epochs=N_EPOCHS) 
de_encoder3_train = de_encoder3.predict(de_encoder2_train)

fig, (a2)  = plt.subplots(1,1, subplot_kw={'projection':'3d'})
a2.scatter(de_encoder3_train[:,0], de_encoder3_train[:,1], de_encoder3_train[:,2], c=t_train, cmap='gist_ncar')
plt.axis('auto')
plt.title('diffusion net decoder output')
plt.show(block=False)


# ## create model ############
# the parameters for training the network
learning_rate = 1e-2
n_iters = 3000

X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, embedding_size])

# params for loss function
reg_par = 1e-4
sessD = tf.Session()
sessE = tf.Session()

######################
####              ####
####    Decoder   ####
######################

print "Train decoder"

# init diffusion net decoder units from pretrained autoencoders
init = tf.constant(de_autoencoder1.get_weights()[0])
D_W1 = tf.Variable(init)
init = tf.constant(de_autoencoder1.get_weights()[1])
D_b1 = tf.Variable(init)
init = tf.constant(de_autoencoder2.get_weights()[0])
D_W2 = tf.Variable(init)
init = tf.constant(de_autoencoder2.get_weights()[1])
D_b2 = tf.Variable(init)
init = tf.constant(de_encoder3.layers[1].get_weights()[0])
D_W3 = tf.Variable(init)

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2]

def decoder(z):
    h1 = tf.nn.sigmoid(tf.matmul(z, D_W1) + D_b1)
    h2 = tf.nn.sigmoid(tf.matmul(h1, D_W2) + D_b2)
    h3 = tf.matmul(h2, D_W3)
    return h3

Y = tf.placeholder(tf.float32, shape=[None, embedding_size])
R = decoder(Y)

decoder_reg = tf.nn.l2_loss(D_W3) #+tf.nn.l2_loss(D_W1) + tf.nn.l2_loss(D_W2)
decoder_loss = tf.reduce_mean(tf.square(X-R)) + reg_par*decoder_reg

D_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate)
            .minimize(decoder_loss, var_list=theta_D))

sessD.run(tf.global_variables_initializer())
#### train decoder
for iter in range(n_iters):
    _ = sessD.run(D_solver, feed_dict={X: S1_train, Y:S1_embedding})

r = sessD.run(R, feed_dict={Y:S1_embedding})

loss = np.mean(np.sum(np.abs(S1_train - r)**2,axis=1)**(1./2))
print('Decoder loss %.2e' % (loss))

fig, (a2)  = plt.subplots(1,1, subplot_kw={'projection':'3d'})
a2.scatter(S1_train[:,0], S1_train[:,1], S1_train[:,2], color='blue', zorder=1)
a2.scatter(r[:,0], r[:,1], r[:,2], color='red', zorder=10)
plt.title('diffusion net decoder output')
plt.show(block=False)
plt.savefig('DN_dec_2layer.png', bbox_inches='tight',transparent = True)

######################
####              ####
####    Encoder   ####
######################
# init diffusion net encoder units from pretrained autoencoders
init = tf.constant(autoencoder1.get_weights()[0])
E_W1 = tf.Variable(init)
init = tf.constant(autoencoder1.get_weights()[1])
E_b1 = tf.Variable(init)
init = tf.constant(autoencoder2.get_weights()[0])
E_W2 = tf.Variable(init)
init = tf.constant(autoencoder2.get_weights()[1])
E_b2 = tf.Variable(init)
init = tf.constant(encoder3.layers[1].get_weights()[0])
E_W3 = tf.Variable(init)

theta_E = [E_W1,E_W2, E_W3, E_b1, E_b2]

def encoder(x):
    h1 = tf.nn. sigmoid(tf.matmul(x, E_W1) + E_b1)
    h2 = tf.nn.sigmoid(tf.matmul(h1, E_W2) + E_b2)
    h3 = tf.matmul(h2, E_W3)
    return h3

# ## Train deep encoder
X = tf.placeholder(tf.float32, shape=[None, input_size])
Z = encoder(X)

# set encoder loss
encoder_fidelity_loss = tf.reduce_mean(tf.square(Y-Z))
encoder_eigen_loss = 0
for i in range(embedding_size):
    mat = P - v1[i]*np.eye(n_train,dtype=np.float32)
    z_vec =  tf.slice(Z,[0,i],[-1,1])
    vec = tf.matmul(mat, z_vec)
    encoder_eigen_loss += tf.reduce_mean(tf.square(vec))
encoder_reg = tf.nn.l2_loss(E_W3) #+tf.nn.l2_loss(E_W1) + tf.nn.l2_loss(E_W2)

fig1 = plt.figure(figsize=(28,4))
fig2 = plt.figure(figsize=(28,4))

eta_vec =[0, 1, 10, 100, 1000,1e5]
for i,eta in enumerate(eta_vec):
    print('eta=%d' % (eta))
    
    encoder_loss = encoder_fidelity_loss + eta*encoder_eigen_loss + reg_par*encoder_reg

    E_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                .minimize(encoder_loss, var_list=theta_E))

    sessE.run(tf.global_variables_initializer())

    z = sessE.run(Z, feed_dict={X: S1_train})
    loss = np.mean(np.sum(np.abs(S1_embedding - z)**2,axis=1)**(1./2))
    print('Initial encoder loss %.2e' % (loss))
    # ### train encoder
    for iter in range(n_iters+1):
        _ = sessE.run(E_solver, feed_dict={X: S1_train, Y:S1_embedding})

    z = sessE.run(Z, feed_dict={X: S1_train})
    loss = np.mean(np.sum(np.abs(S1_embedding - z)**2,axis=1)**(1./2))
    print('Final encoder loss %.2e' % (loss))

    # net outputs:

    a1 = fig1.add_subplot(1,len(eta_vec),i+1)
    a1.scatter(S1_embedding[:,0], S1_embedding[:,1],color='blue', zorder=1)
    a1.scatter(z[:,0], z[:,1],color='red', zorder=10)
    a1.set_aspect('equal')
    a1.set_title('eta=' +str(eta))

    ### Stack autoencoder
    z_test = sessE.run(Z, feed_dict={X: S1_test})
    r_test = sessD.run(R, feed_dict={Y:z_test})
    # compare loss to original *clean* data
    loss = np.mean(np.sum(np.abs(clean_test - r_test)**2,axis=1)**(1./2))
    print('Full autoencoder denoising loss %.2e' % (loss))

    a2 = fig2.add_subplot(1,len(eta_vec),i+1,projection='3d')
    a2.scatter(r_test[:,0], r_test[:,1], r_test[:,2], color='red')
    a2.set_title('eta=' +str(eta))
    a2.set_aspect('equal')


plt.show(block=False)
fig1.savefig('DN_enc_2layer.png', bbox_inches='tight',transparent = True)

fig2.savefig('DN_stack_2layer.png', bbox_inches='tight',transparent = True)





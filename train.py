import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import librosa

os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.GPUOptions(allow_growth = True)
config=tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.34
#No. of files

n_file = 1200

max_length = 600

sr = 16000
#Loading the file, performing STFT, taking absolute, padding zeros as required

def loadfile0(path, flag = 0):
    list_tr = []
    list_stft = []
    list_stft_abs = []
    list_length = []
    f_list = os.listdir(path)
    for j in range(1, 501):
        i = str(j) + '.wav'
        s, sr = librosa.load(path + i, sr = None)

        print(i)
        if (flag == 1):
            list_tr.append(s)
        
        #Calculating STFT
        stft = librosa.stft(s, n_fft= 1024, hop_length= 512)
        
        stft_len = stft.shape[1]
        
        #Appending STFT to list
        if (flag == 1):
            list_stft.append(stft)
        
        #Calculating Absolute of STFT
        stft_abs = np.abs(stft)
        
        #Padding zeros to make length 300
        stft_abs = np.pad(stft_abs, ((0,0),(0, max_length-stft_len)), 'constant')
        
        #Appending abs to list
        list_stft_abs.append(stft_abs)
        
        #Appending time-length of STFT to list
        list_length.append(stft_len)
        
    return list_tr, list_stft, list_stft_abs, list_length

  #Path of the training signals
x_path = "/content/drive/My Drive/train/mixed/"
n_path = "/content/drive/My Drive/train/noise/"
s_path = "/content/drive/My Drive/train/speech/"

#Loading all training noisy speech signals

trx, X, X_abs, X_len = loadfile0(x_path)

#Loading all training clean speech signals

trs, S, S_abs, S_len = loadfile0(s_path)

#Loading all training noise signals

trn, N, N_abs, N_len = loadfile0(n_path)

def IBM(S, N):
    M = []
    
    for i in range(len(S)):
        m_ibm = 1 * (S[i] > N[i])
        M.append(m_ibm)
    
    return M

#Getting Binary Masks from S_abs and N_abs
M = IBM(S_abs, N_abs)

batch_size = 10

#Keep probability for dropouts
keep_pr = tf.placeholder(tf.float32, ())
#Placeholders for input to the network

frame_size = 513
num_hidden = 256
seq_len = tf.placeholder(tf.int32, None)

q2_x = tf.placeholder(tf.float32, [None, max_length, frame_size])
q2_y = tf.placeholder(tf.float32, [None, max_length, frame_size])

#Defining the RNN
output, state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_hidden, 
                                                         initializer = tf.contrib.layers.xavier_initializer()),
                                                                output_keep_prob = keep_pr), 
                                  q2_x, dtype=tf.float32, sequence_length=seq_len)

rnn_out = tf.layers.dense(output, 513, kernel_initializer= tf.contrib.layers.xavier_initializer())

dim = seq_len[0]

fin_out = tf.sigmoid(rnn_out)

lr = 0.001

cost = tf.reduce_mean(tf.losses.mean_squared_error(fin_out[:, :dim,:], q2_y[:, :dim, :]))

optimizer = tf.train.AdamOptimizer(learning_rate= lr).minimize(cost)
#Init all TF vars and run the session

sess = tf.Session(config = config)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

epochs = 200
error = np.zeros(epochs)
#Training the network with shuffling of training batches


for epoch in range(epochs):
    random = np.arange(0, 500, 10)
    np.random.shuffle(random)
    for i in range(len(random)):
        start = int(random[i])
        end = int(start + batch_size)
        epoch_y = np.array(M[start:end]).swapaxes(1,2)
        epoch_x = np.array(X_abs[start:end]).swapaxes(1,2)
        seqlen = np.array(X_len[start:end])
        l, _ = sess.run([cost, optimizer], feed_dict = {q2_x: epoch_x, q2_y: epoch_y, seq_len: seqlen, keep_pr: 1})
        error[epoch] += l
    
    print('Epoch', epoch+1, 'completed out of ', epochs,'; loss: ', error[epoch])

saver.save(sess, 'q3model/q2')

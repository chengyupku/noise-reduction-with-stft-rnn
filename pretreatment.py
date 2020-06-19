import librosa
import numpy as np
import os

SNR = 0.5 #signal noise ratio

def add_noise(data):
    wn = np.random.normal(0, 1, len(data))
    noise_rate = np.sqrt(SNR * np.var(data))
    wn_ = np.where(data != 0.0,  noise_rate * wn, 0.0).astype(np.float32)
    data_noisy = np.where(data != 0.0, data.astype('float64') + noise_rate * wn, 0.0).astype(np.float32)
    return wn_, data_noisy

def generate_file(file_path, myfolder = None, rename = None):
    folder, file_name = os.path.split(file_path)
    if myfolder is None:
        myfolder = folder
    if rename is None:
        rename = file_name
    data, fs = librosa.core.load(file_path)
    wn, data_noisy = add_noise(data)
    pathN = os.path.join(myfolder, 'noise',rename)
    librosa.output.write_wav(pathN, wn, fs)
    pathX = os.path.join(myfolder, 'mixed' , rename)
    librosa.output.write_wav(pathX, data_noisy, fs)
    pathS = os.path.join(myfolder, 'speech' , rename)
    librosa.output.write_wav(pathS, data, fs)

rootdir = 'TIMIT/TEST'
num = 0
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(len(list)):
    path0 = os.path.join(rootdir,list[i])
    #print(path)
    list1 = os.listdir(path0)
    for j in range(len(list1)):
        path = os.path.join(path0, list1[j])
        list2 = os.listdir(path)
        for k in range(len(list2)):
            _, type = os.path.splitext(list2[k])
            if type == '.WAV':
                num = num + 1
                file_path = os.path.join(path ,list2[k])
                print(file_path)
                generate_file(file_path, myfolder='test1',rename=str(num)+'.wav')





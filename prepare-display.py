import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import keras
import os
import time

def fix_nbr_of_points(X,Y,n_points,x_min,x_max):
    """
    Reshapes the curve to decrease or increase its number of points (n points, between x_min and x_max) 
 
    Parameters
    ----------
    X : list of floats
        time/frequency
    Y : list of floats
        volatage of the signal
    n_points : int
        number of desired points
    x_min : int
        ?
    x_max : int
        ?
        
    Returns
    ----------
    X2 : lists of floats 
        final time/frequency curve 
    Y2 : lists of floats 
        final voltage curve 
    """
    x = x_min
    X2 = []
    Y2 = []
    e = np.mean([X[k+1]-X[k] for k in range(len(X)-1)])
    i = 0
    while(i<n_points):
        i+=1
        X2.append(x)
        k = (x-X[0])/e
        k_int = int(k)
        if(k>=len(X)-1):
            Y2.append(Y[k_int])
        else:
            Y2.append((Y[k_int]+ (k-k_int)*(Y[k_int+1]-Y[k_int])))
        x += (x_max-x_min)/(n_points-1)
    return(X2,Y2)

def analyse_slowly(name):
    """
    Analyses a csv file and uses the AI model to predict a broken fan or not, considering the type of failure
    Saves all the signals, fft plots as .png files and all the AI results in a .txt called results.txt
    
    Parameters
    ----------
    name : str
        name of the file
    """
    model = keras.models.load_model(os.getcwd() + "\\data\\model",compile=False)
    n_inputs = model.input.shape[1]
    with open(os.getcwd()+"\\data\\"+name, newline='') as f:
        reader = csv.reader(f)
        data = [tuple(row) for row in reader]
    j=0
    print('Currently working ',end='')
    with open('data/results.txt','w') as f:
        f.write('')
    
    folder = os.getcwd()+'\\images\\curves'
    
    for filename in os.listdir(folder):
        if filename.endswith('.png') and '_0' not in filename and '_1' not in filename:
            file_path = os.path.join(folder, filename)
            os.remove(file_path)
    
    t=time.time()
    while j<=len(data)-50301:
        X,Y=[],[]
        for i in range(j,j+50000):
            (x,y)=data[i]
            x=float(x)
            y=float(y)
            X.append(x)
            Y.append(y)
        
        X,Y=np.array(X),np.array(Y)
        Y=Y-np.mean(Y)
        SAMPLE_RATE= np.mean([X[k+1]-X[k] for k in range(len(X)-1)])
        N = len(X)
        
        plt.plot(X,Y,linewidth=0.4)
        plt.title('Voltage versus time for '+name)
        plt.ylabel('Voltage(V)')
        plt.xlabel('Time(s)')
        plt.savefig(os.path.join(os.getcwd()+"\\images\\curves\\", "courbe_"+str(j//300)+".png"))
        plt.clf()
        
        yf = fft(Y)
        yf = np.abs(np.real(yf))
        xf = fftfreq(N, SAMPLE_RATE)
        xf,yf = xf[0:len(xf)//30],yf[0:len(yf)//30]
        
        plt.plot(xf,yf,linewidth=0.4)
        plt.ylim([0,350])
        plt.title('FFT in real time '+name)
        plt.ylabel('Voltage(V)')
        plt.xlabel('Frequency(Hz)')
        plt.savefig(os.path.join(os.getcwd()+"\\images\\curves\\", "fft_"+str(j//300)+".png"))
        plt.clf()
        
        xf,yf = fix_nbr_of_points(xf.tolist(),yf.tolist(),n_inputs,0,200)
        resultat = model.predict([yf],verbose = 0)[0][0]
        with open('data/results.txt','a') as f:
            f.write(str(resultat)+'\n')
        if (j//300)%5==0:
            print('=',end='')
        j+=300
    print(time.time()-t)
        
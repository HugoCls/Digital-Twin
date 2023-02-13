import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.fft import fft, fftfreq
import pandas as pd
import keras
import tensorflow as tf
import os
from openpyxl import load_workbook, Workbook

categories = ['new', 'unbalanced', 'scratched']


def xlsx_to_csv(path):
    """
    Changing an xlsx file to a csv file

    Parameters
    ----------
    path : str
        relative path of the file

    """
    wb = load_workbook(os.getcwd()+"\\data\\" + path)
    ws = wb['Sheet1']
    ws.delete_rows(1,2)
    ws.delete_cols(1,1)
    wb.save(os.getcwd()+"\\data\\" + path)
    read_file = pd.read_excel(os.getcwd()+"\\data\\" + path)
    read_file.to_csv(os.getcwd()+"\\data\\" + path[:-4] + "csv")
    os.remove(os.getcwd()+"\\data\\" + path)

def folder_xlsx_to_csv():
    """
    Changing all xlsx files to a csv file in subfolders
    
    """
    for path in os.listdir(os.getcwd()):
        if(path.endswith(".xlsx")):
            print(path)
            xlsx_to_csv(path)

def random_strip(N,X,Y):
    """
    Creation of two sub-lists of length N of the two previous lists

    Parameters
    ----------
    N : int
        len of the sublists
    X : list of floats
        time/frequency
    Y : list of floats
        voltage of the signal
        
    Returns
    ----------
    X' : lists of floats 
        sublist of X 
    Y' : lists of floats 
        sublist of Y with the same corresponding indices
        
    """
    L = len(X)
    x0 = np.random.randint(0,L+1-N)
    return(X[x0:x0+N],Y[x0:x0+N])


def fill_in(curves):
    """
    Creation of the desired signals, mutiplying the amount of data

    Parameters
    ----------
    curves : list of float lists
        initial curves
    """
    for name in([("test23_",[1,0,0]),("test24_",[0,1,0]),("test25_",[0,1,0]),("test29_",[0,0,1]),("test31_",[0,0,1]),("test33_",[0,0,1]),("test35_",[0,0,1]),("test39_",[0,0,1]),("test37_",[0,0,1])]):
        for j in range(1,5):
            print(j)
            with open(os.getcwd()+"\\data\\"+name[0]+str(j)+'.csv', newline='') as f:
                reader = csv.reader(f)
                data = [tuple(row) for row in reader]
            X,Y=[],[]

            for i in range(2,len(data)):
                if(len(data[i])> 2):
                    data[i] = data[i][-2:]
                (x,y)=data[i]
                x=float(x)
                y=float(y)
                X.append(x)
                Y.append(y)

            X,Y=np.array(X),np.array(Y)
            Y=Y-np.mean(Y)
            # Number of samples in normalized_tone

            for i in range(30):
                X2,Y2 = random_strip(len(X)//2,X,Y)
                SAMPLE_RATE= np.mean([X2[k+1]-X2[k] for k in range(len(X2)-1)])
                DURATION= X2[len(X2)-1]-X[0]
                N = len(X2)
                yf = fft(Y2)
                yf = np.abs(yf)
                xf = fftfreq(N, SAMPLE_RATE)
                xf,yf = xf[0:len(xf)//2],yf[0:len(yf)//2]
                curves.append(xf)
                curves.append(yf)
                curves.append(name[1])

def fix_number_of_points(X,Y,n_points,x_min,x_max):
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
        the number of points of the smallest curve
    x_max : int
        the number of points of the largest curve
        
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


def nb_point_to(L,x_max):
    k = 1
    while(L[k-1]<x_max and k<len(L)-1):
        k+=1
    return(k)

def normalise(curves):
    n_max = nb_point_to(curves[0],200)
    couleurs = ['b','g','r','m']
    for k in range(len(curves)//3):
        (curves[3*k],curves[3*k+1]) = fix_number_of_points(curves[3*k].tolist(),curves[3*k+1].tolist(),n_max,np.min(curves[3*k]),200)

def create_model(curves = []):
    """
    Creation of the model used for AI analysis

    Parameters
    ----------
    curves : list of prepared signals
        
    Returns
    ----------
    model : keras.Sequential class
        the model itself
    history : Model.history class
    
    """
    if(len(curves) == 0):
        fill_in(curves)
        normalise(curves)
    n_max = max([len(curves[3*k]) for k in range(len(curves)//3)])
    print(n_max)
    LAYERS = [keras.layers.Input(n_max,dtype="float32")]
    for k in range(3):
        LAYERS.append(keras.layers.Dense((64//(2**k)),activation="relu",dtype="float32"))
    LAYERS.append(keras.layers.Dense(len(curves[2]),activation=tf.keras.activations.softmax))
    model = tf.keras.Sequential(LAYERS)
    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='mse',
        metrics=[tf.keras.metrics.CategoricalAccuracy()])
    history = model.fit(
        np.array([curves[3*k+1] for k in range(len(curves)//3)]),
        np.array([curves[3*k+2] for k in range(len(curves)//3)]),
        epochs=10,
        verbose=1,
        validation_split = 0.2)
    return(model,history)


def find_model(curves = []):
    """
    Creation of models with random start until the satisfactory model is obtained to avoid the problem of local maximums

    Parameters
    ----------
    curves : list of prepared signals
        
    Returns
    ----------
    accuracy_max : float between 0 and 1
        the best accuracy found
    m_actual : keras.Sequential class
        the chosen model
    
    h_actual : Model.history class
    """
    if(len(curves) == 0):
        fill_in(curves)
        normalise(curves)
    accuracy_max = 0
    m_actual = None
    h_actual = None
    k = 0

    while(accuracy_max < 0.8 and k<200):
        print(k)
        k+=1
        (m,h) = create_model(curves)
        if(h.history["categorical_accuracy"][9]>accuracy_max):
            accuracy_max = h.history["categorical_accuracy"][9]
            m_actual = m
            h_actual = h
    return(accuracy_max,m_actual,h_actual)

def analyse(file_name):
    """
    Analyses a csv file and uses the AI model to predict a broken fan or not, considering the type of failure
    
    Parameters
    ----------
    file_name : str
        name of the file
    Returns
    ----------
    result : list of 3 floats between 0 and 1
        percentage of apartaining to a certain class for each breakage class
    """
    path=os.getcwd()+"\\data\\"+file_name
    model = keras.models.load_model(os.getcwd() + "\\data\\model\\multi-categorical-AI")
    n_inputs = model.input.shape[1]
    n_outputs = model.output.shape[1]
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = [tuple(row) for row in reader]
    X,Y=[],[]

    for i in range(2,len(data)):
        (x,y)=data[i][-2:]
        x=float(x)
        y=float(y)
        X.append(x)
        Y.append(y)
    L = []
    for k in range(50):
        X2 = X.copy()
        Y2 = Y.copy()
        X2,Y2 = random_strip(len(X2)//2,X2,Y2)
        X2,Y2=np.array(X2),np.array(Y2)
        Y2=Y2-np.mean(Y2)
        SAMPLE_RATE= np.mean([X2[k+1]-X2[k] for k in range(len(X2)-1)])
        DURATION= X2[len(X2)-1]-X2[0]
        N = len(X2)
        yf = fft(Y2)
        yf = np.abs(np.real(yf))
        xf = fftfreq(N, SAMPLE_RATE)
        xf,yf = xf[0:len(xf)//30],yf[0:len(yf)//30]
        xf,yf = fix_number_of_points(xf.tolist(),yf.tolist(),n_inputs,0,200)
        L.append(yf)
    plt.plot(xf,yf,linewidth=0.4)
    plt.savefig(os.getcwd()+"\\images\\analysed.png")
    #plt.show()
    predict = model.predict(L)
    result = [0] * n_outputs
    for k in range(50):
        for n in range(n_outputs):
            result[n] += predict[k][n] / 50
    print("This fan is", categories[result.index(max(result))])
    return(result)
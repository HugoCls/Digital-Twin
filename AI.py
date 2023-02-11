import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import keras
import tensorflow as tf
import openpyxl
import os

def random_tranche(N,X,Y):
    L = len(X)
    x0 = np.random.randint(0,L+1-N)
    return(X[x0:x0+N],Y[x0:x0+N])

def fill_in(curves):
    for nom in([("casse1_",0),("neuf1_",1)]):
        for j in range(1,6):
            print(j)
            with open(os.getcwd()+'\\data\\'+nom[0]+str(j)+'.csv', newline='') as f:
                reader = csv.reader(f)
                data = [tuple(row) for row in reader]
            X,Y=[],[]

            for i in range(2,len(data)):
                (x,y)=data[i]
                x=float(x)
                y=float(y)
                X.append(x)
                Y.append(y)

            X,Y=np.array(X),np.array(Y)
            Y=Y-np.mean(Y)
            # Number of samples in normalized_tone

            for i in range(1000):
                X2,Y2 = random_tranche(len(X)//2,X,Y)
                SAMPLE_RATE= np.mean([X2[k+1]-X2[k] for k in range(len(X2)-1)])
                #DURATION= X2[len(X2)-1]-X[0]
                N = len(X2)
                yf = fft(Y2)
                yf = np.abs(np.real(yf))
                xf = fftfreq(N, SAMPLE_RATE)
                xf,yf = xf[0:len(xf)//30],yf[0:len(yf)//30]
                curves.append(xf)
                curves.append(yf)
                curves.append(nom[1])

def plus_proche(X,Y,x_cible,e):
    k = int(np.round((x_cible-X[0])/e))
    if(k < 0):
        print("trop bas")
        k = 0
    if(k>=len(X)):
        k = len(X)-1
    return(Y[k])


def diff_fft(k1,k2,C):
    y1 = C[3*k1+1]
    y2 = C[3*k2+1]
    x1 = C[3*k1]
    x2 = C[3*k2]
    e1 = x1[1]-x1[0]
    e2 = x2[1]-x2[0]
    x = min(x1)
    x_max = max(x1)
    s = 0
    k = 0
    if(e1<e2):
        while(x < x_max):
            s+= abs(y1[k] - plus_proche(x2,y2,x,e2))
            k =+ 1
            x += e1
    else:
        while(x < x_max):
            s+= abs(y2[k] - plus_proche(x1,y1,x,e1))
            k =+ 1
            x += e2
    return(s)

def fix_number_of_points(X,Y,n_points,x_min,x_max):
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



def normalise(curves):
    n_max = max([len(curves[3*k]) for k in range(len(curves)//3)])
    colors = ['b','g','r','m']
    for k in range(len(curves)//3):
        (curves[3*k],curves[3*k+1]) = fix_number_of_points(curves[3*k].tolist(),curves[3*k+1].tolist(),n_max,np.min(curves[3*k]),200)
        plt.plot(curves[3*k],curves[3*k+1],colors[curves[3*k+2]])
    plt.show()


def create_model(curves = []):
    if(len(curves) == 0):
        fill_in(curves)
        normalise(curves)
    n_max = max([len(curves[3*k]) for k in range(len(curves)//3)])


    LAYERS = [keras.layers.Input(n_max,dtype="float32")]
    for k in range(3):
        LAYERS.append(keras.layers.Dense((64//(2**k)),activation="relu",dtype="float32"))
    LAYERS.append(keras.layers.Dense(1,activation="sigmoid"))
    model = tf.keras.Sequential(LAYERS)
    model.summary()
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='mse',
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    history = model.fit(
        np.array([curves[3*k+1] for k in range(len(curves)//3)]),
        np.array([curves[3*k+2] for k in range(len(curves)//3)]),
        epochs=10,
        verbose=1,
        validation_split = 0.2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    L_estime = model.predict([curves[3*k+1] for k in range(len(curves)//3)]).tolist()

    err = 0
    for k in range(len(curves)//3):
        if(k < len(curves)//6 and L_estime[k][0]>0.5):
            err += 1
        if(k>= len(curves)//6 and L_estime[k][0]<0.5):
            err += 1
            
    print(err," error, on ",len(curves)//3, " exemples")
    model.save("mymodel")
    
    return(err,model,history)



def analyse(lien):
    model = keras.models.load_model(os.getcwd() + "\\data\\model")
    n_inputs = model.input.shape[1]
    if lien[-3:]=="csv":
        with open(lien, newline='') as f:
            reader = csv.reader(f)
            data = [tuple(row) for row in reader]
        X,Y=[],[]
        for i in range(2,len(data)):
            (x,y)=data[i]
            x=float(x)
            y=float(y)
            X.append(x)
            Y.append(y)
            
        X,Y=np.array(X),np.array(Y)
 
    else:
        wb = openpyxl.load_workbook(lien)
        ws = wb.active
        X,Y=[],[]
        for row in ws.iter_rows(values_only=True):
            if row[0]!=None and row[0]>=1:
                (x,y)=(row[1],row[2])
                #print(x,y)
                x=float(x)
                y=float(y)
                X.append(x)
                Y.append(y)
                
        X,Y=np.array(X),np.array(Y)

    X,Y=np.array(X),np.array(Y)
    Y=Y-np.mean(Y)
    SAMPLE_RATE= np.mean([X[k+1]-X[k] for k in range(len(X)-1)])
    #DURATION= X[len(X)-1]-X[0]
    N = len(X)
    yf = fft(Y)
    yf = np.abs(np.real(yf))
    xf = fftfreq(N, SAMPLE_RATE)
    xf,yf = xf[0:len(xf)//30],yf[0:len(yf)//30]
    plt.plot(xf,yf,linewidth=0.4)
    plt.savefig("new_img.png")
    xf,yf = fix_number_of_points(xf.tolist(),yf.tolist(),n_inputs,0,200)
    resultat = model.predict([yf])[0][0]
    return(resultat)

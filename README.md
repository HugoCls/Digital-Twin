# Digital-Twin
---
## Our project and the IT challenges

### The context

The aim of our project is to be able to **anticipate breakdowns** in **industrial ventilation**.

For this we use a hot wire which acts as an anemometric sensor. When air arrives on the wire, so that its temperature remains constant, a certain **voltage** is delivered and it is this one that we study.

Using this voltage only, we want to know what state the fans in the duct are/will be in.

### Retrieve the data
The acquisition was done using a dantec and we recovered .csv files that we could then use

### Data processing
We can easily get a list of tuples containing (t,U) as here `data=[('0.0001', '1.0684'), ... ,('0.0999', '1.0523'), ('0.1', '1.0602')]`.

With `csv.reader`function:

```Python
with open(os.getcwd()+"\\data\\"+name, newline='') as f:
        reader = csv.reader(f)
        data = [tuple(row) for row in reader]
```
Then we put these on two lists `X, Y`.

We then used `scipy.fft`, `scipy.fftfreq` to get the FFT of the signal.

```Python
SAMPLE_RATE= np.mean([X[k+1]-X[k] for k in range(len(X)-1)])
N = len(X)
yf = fft(Y)
xf = fftfreq(N, SAMPLE_RATE)
```
Such parameters are based on the number of points in the file because we had some issues with our acquisition card which sometimes made acquisitions with non-constant numbers of points.

We can easily see now that when comparing both ffts of a broken fan and a clean one,  we can see new frequences 
![My Image](images/README_IMAGES/differencing_ffts.png)

And when a blade is removed from the fan, we most of the time see frenquency lines at 57Hz:

![My Image](images/README_IMAGES/zoom_on_broken_ffts.png)

Unfortunately, this is not at simple as that and knowing that degradations do not happen suddenly and as our role is to predict themand not to detect them, we decided to use an artificial intelligence model. 

## The place of AI in the project

The principle is to present cases of degradation to a neural network so that it can train itself and then anticipate a type of degradation.

### Train a model
***
For the first AI model we wanted something simple: You enter a curve (points list) and you get as a binary output, if the fan was broken/damaged or not.

#### Choice of model type

Our first LAYER is an **input** with the **number of points of an ascquisition**.
```Python
LAYERS = [keras.layers.Input(n_max,dtype="float32")]
```
Then we add some LAYERS to train:
```Python
for k in range(3):
    LAYERS.append(keras.layers.Dense((64//(2**k)),activation="relu",dtype="float32"))
```

Finally ,the last LAYER had to be a sigmoid to get our **binary output**.

```Python
LAYERS.append(keras.layers.Dense(1,activation="sigmoid"))
```

```Python
model = tf.keras.Sequential(LAYERS)
model.summary()
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='mse',
    metrics=[tf.keras.metrics.BinaryAccuracy()])

```
**Adam** is a classical choice for *Stochastic optimisation*.(the model makes random oriented modifications that it chooses according to the results it obtains on its own calculations)

About `mse` loss choice: **MSE** ensures that our trained model does not have outliers with huge errors, as MSE gives more weight to these errors due to the squared function.

Disadvantage: if our model makes a single very bad prediction, the squared part of the function amplifies the error. 

In our case it is to be avoided that a broken fan is not detected as such, so the MSE error seems very relevant 


#### Mutiply the amount of data



#### Generate the model
```Python
def create_model(curves = []):
    if(len(curves) == 0):
        remplir(curves)
        normaliser(curves)
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
    print(err," erreur, sur ",len(curves)//3, " exemples")
    model.save("mymodel")
    return(err,model,history)
```

### Model performance

![My Image](images/README_IMAGES/model_loss.png)

## The final product with graphics

![My Image](images/README_IMAGES/gui_exemple.png)


## Exemple
Here your can see some demonstrations on the thing in action:

   - [1st version of the app](https://www.youtube.com/watch?v=_6Yb9YLgItU&list=PL_7_H9j4EBUP2sV3jfq105vpyNiUSdNKO&index=3 "Digital-Twin application.")
   
   
   - [2d version of the app](https://youtu.be/pZooMAW_KgA")



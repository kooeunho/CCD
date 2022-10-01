import numpy as np
import tensorflow as tf

def unit(x):
    return (x - np.min(x,0))/(np.max(x,0) - np.min(x,0))

def RMSE(x,y):
    return np.sqrt( ( ( x.flatten() - y.flatten() )**2 ).mean() )

def tf_RMSE(y, y_hat):
    return tf.sqrt(tf.reduce_mean(tf.square(y - y_hat)))

def CC(x,y):
    return np.corrcoef(x.flatten(), y.flatten())[0,1]

def extreme(Prediction, Label, ratio):
    mat = np.concatenate([Prediction.reshape(-1,1), Label.reshape(-1,1)], axis=1)
    mat1 = mat[mat[:,1].argsort()][::-1]
    position = int(ratio * len(mat))
    threshold_value = mat1[position][1]
    Mat = []
    for i in range(len(mat)):
        if mat[i,1] >= threshold_value:
            Mat.append(mat[i])
    consider = np.array(Mat)        
    new_pred, new_label = consider[:,0], consider[:,1]        
    return RMSE(new_pred, new_label)

def weist(data, alpha):
    def unit(x):
        return (x - np.min(x))/(np.max(x) - np.min(x))
    data = unit(data)
    beta = 0.5**(1-alpha)
    Data = []
    for i in range(len(data)):
        if data[i] < 1/2:
            Data.append( beta * (data[i]**alpha) )
        else:
            Data.append( ( (1/beta)**(1/alpha) ) * ( (data[i]-0.5)**(1/alpha) ) + 0.5)
    return np.array(Data)

def inv_weist(prediction, original_data, alpha):
    beta = 0.5/((0.5)**alpha)
    M, N = np.max(original_data), np.min(original_data)
    pred = []
    for i in range(len(prediction)):
        if prediction[i] < 1/2:
            pred.append( (1/beta*prediction[i])**(1/alpha) )
        else:
            pred.append( beta*( (prediction[i]-0.5)**(alpha) ) + 0.5  )
    return np.array(pred)

def gaussian_pdf(mean, std):
    def Gaussian(x):
        pdf = (1/(2*np.pi*std**2)**(1/2)) * np.exp(-((x - mean)**2)/(2*std**2))
        return(pdf)
    return(Gaussian)

def kde_pdf(kernel_function, label, std):
    def PDF(x):
        pdf = sum([ kernel_function(label[i], std)(x) for i in range(len(label)) ])/len(label)
        return pdf
    return PDF 

def error_factor(x, pdf, label, beta, Type):
    X = np.linspace(np.min(label), np.max(label), 1000)
    M, N = np.max(pdf(X)), np.min(pdf(X))
    if Type == 0:
        return np.ones(len(x))
    if Type == 1:
        return (beta - 1)*(-( (pdf(x) - N) / (M - N) ) + 1) + 1    
    
def split_LSTM(data, label, depth, ratio):
    x_data     = data[:-1]
    y_data     = label[depth:]

    x_setting = []
    for i in range(depth-1):
        x_setting.append(x_data[i:-(depth-i-1)])
    x_setting.append(x_data[depth-1:])
    y_setting = y_data

    ntrain = int(ratio*len(x_setting[0]))
    trainX = [ x_setting[i][:ntrain] for i in range(depth) ]
    trainY = y_setting[:ntrain].reshape(-1,1)
    testX  = [ x_setting[i][ntrain:] for i in range(depth) ]
    testY  = y_setting[ntrain:].reshape(-1,1)
    return trainX, trainY, testX, testY

def train_batch_LSTM(trainX, trainY, depth, batch_size):
    if len(trainX[0]) % batch_size == 0:
        L = int(len(trainX[0]) / batch_size)
    else:
        L = int(len(trainX[0]) / batch_size) + 1
        
    TrainX, TrainY = [ [] for i in range(L) ], []
    for i in range(L):
        TrainY.append(trainY[i*batch_size : (i+1)*batch_size])
    for dep in range(depth):
        for i in range(L):
            TrainX[i].append(trainX[dep][i*batch_size : (i+1)*batch_size])  
    return TrainX, TrainY 

def LSTM(trainX, trainY, testX, testY, depth, hidden_dim, lr1, lr2, batch_size, epochs, error_f):
    
    tf.reset_default_graph()
    input_dim  = trainX[0].shape[1]
    output_dim = trainY.shape[1]

    activation1 = tf.nn.sigmoid
    activation2 = tf.math.tanh
    #activation3 = elu
    
    us = 1/np.sqrt(5)
    
    TrainX, TrainY = train_batch_LSTM(trainX, trainY, depth, batch_size)
    
    X = [tf.placeholder(tf.float32, [None,  input_dim]) for i in range(depth)]
    Y =  tf.placeholder(tf.float32, [None, output_dim]) 

    Weight1 = tf.Variable(tf.random_uniform([input_dim , 4*hidden_dim], -us, us))
    Weight2 = tf.Variable(tf.random_uniform([hidden_dim, 4*hidden_dim], -us, us))
    bias    = tf.Variable(tf.random_uniform([            4*hidden_dim], -us, us))

    W = tf.Variable(tf.random_uniform([hidden_dim, 1], -us, us))
    b = tf.Variable(tf.random_uniform([            1], -us, us))

    A = tf.matmul(X[0], Weight1) + bias
    F, I ,O, G, C, H = [],[],[],[],[],[]
    F.append(activation1(A[:,              : 1*hidden_dim]))
    I.append(activation1(A[:, 1*hidden_dim : 2*hidden_dim]))
    O.append(activation1(A[:, 2*hidden_dim : 3*hidden_dim]))
    G.append(activation2(A[:, 3*hidden_dim :             ]))
    C.append( I[-1] * G[-1] )
    H.append( O[-1] * activation2( C[-1] ))
    for i in range(1, depth):
        B = tf.matmul(X[i], Weight1) + tf.matmul(H[-1], Weight2) + bias    
        F.append(activation1(B[:,              : 1*hidden_dim]))
        I.append(activation1(B[:, 1*hidden_dim : 2*hidden_dim]))
        O.append(activation1(B[:, 2*hidden_dim : 3*hidden_dim]))
        G.append(activation2(B[:, 3*hidden_dim :             ]))
        C.append( F[-1] * C[-1] + I[-1] * G[-1] )
        H.append( O[-1] * activation2( C[-1] )) 
      
    output = tf.matmul(H[-1], W) + b   
    cost = tf.reduce_mean(tf.square(Y - output)*error_f) 
    
    lr   = tf.placeholder(tf.float32, [])
    gogo = tf.train.AdamOptimizer(lr).minimize(cost)
    
    real = tf.placeholder(tf.float32, [None, output_dim]) 
    pred = tf.placeholder(tf.float32, [None, output_dim]) 
    rmse = tf_RMSE(real, pred)  
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    LR = lr1*np.exp(-1/epochs*np.log(lr1/lr2)*np.linspace(0,epochs,epochs)) 
    # exponential decay from lr1 to lr2
    
    for epoch in range(epochs):
        for i in range(len(TrainX)):
            feed1 = {Y:TrainY[i], lr:LR[epoch]}
            for dep in range(depth):
                feed1[X[dep]] = TrainX[i][dep]                  
            sess.run(gogo, feed_dict = feed1)
            
        if epoch % int(epochs/5) == 0:  
            feed2 = {}
            for dep in range(depth):
                feed2[X[dep]] = testX[dep]
            training_error = sess.run(tf.sqrt(cost),  feed_dict = feed1)
            prediction     = sess.run(output       ,  feed_dict = feed2) 
            testing_error  = sess.run(rmse,  feed_dict = {real:testY, pred:prediction})    
            print('Epoch:', epoch, 'Training Error:',training_error,'and','Testing Error:', testing_error)
          
    return prediction.flatten()
















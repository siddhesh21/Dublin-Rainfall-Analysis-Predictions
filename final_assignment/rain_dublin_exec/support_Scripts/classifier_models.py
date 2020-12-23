import warnings
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.python.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
import pickle
from sklearn.metrics import accuracy_score
import support_Scripts.pre_process

def logistic_regression_model(X_train, y_train):
    logisticReg = LogisticRegression(penalty="l1",max_iter = 300, solver = "liblinear") #max_iter alteration for different values
    logisticReg.fit(X_train, y_train)
    y_pred_logistic = logisticReg.predict(X_train)
    logisticRegression_accuracy = metrics.accuracy_score(y_train,y_pred_logistic)
    print("LOGISTIC MODEL ACCURACY OVER TRAINING DATA : ",logisticRegression_accuracy)
    
    filename = '../Models/logistic_regression.sav'
    pickle.dump(logisticReg, open(filename, 'wb'))
    print("LOGISTIC REGRESSION MODEL SAVED IN MODELS FOLDER.")

def SVM_model(X_train, y_train):
    #Chosen model with best suited C value.
    C_value = 10
    svmModel = svm.SVC(C = C_value, kernel = "rbf")
    svmModel.fit(X_train,y_train)
    print("SVM MODEL ACCURACY OVER TRAINING DATA : ",svmModel.score(X_train,y_train))
    
    filename = '../Models/SVM.sav'
    pickle.dump(svmModel, open(filename, 'wb'))
    print("SVM MODEL SAVED IN MODELS FOLDER.")

def kNN_model(X_train, y_train):
    kNNmodel = KNeighborsClassifier(n_neighbors=4)
    kNNmodel.fit(X_train, y_train)
    y_pred_kNN = kNNmodel.predict(X_train)
    kNN_accuracy = accuracy_score(y_train,y_pred_kNN)
    print("kNN MODEL ACCURACY OVER TRAINING DATA : ",kNN_accuracy)
    
    filename = '../Models/kNN.sav'
    pickle.dump(kNNmodel, open(filename, 'wb'))
    print("kNN MODEL SAVED IN MODELS FOLDER.")

def ridgeModel(X_train, y_train):
    C_value = 10
    ridge_model = RidgeClassifier(alpha = 1/(2*C_value))
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_train)
    ridge_accuracy = accuracy_score(y_train,y_pred_ridge)
    print("RIDGE CLASSIFIER MODEL ACCURACY OVER TRAINING DATA : ",ridge_accuracy)
    
    filename = '../Models/ridge.sav'
    pickle.dump(ridge_model, open(filename, 'wb'))
    print("RIDGE MODEL SAVED IN MODELS FOLDER.")

def neuralNetwork_model(X_train, y_train):
    nnmodel = Sequential()
    nnmodel.add(Dense(16, activation='relu', input_shape=(6,)))
    nnmodel.add(Dense(16, activation='relu'))
    nnmodel.add(Dense(16, activation='relu'))
    nnmodel.add(Dropout(0.25))
    nnmodel.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1(0.0001)))

    nnmodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    nnmodel.fit(X_train,y_train,epochs=20, batch_size=64, validation_split=0.1)

    # y_pred_nn = nnmodel.predict_classes(X_train)
    y_pred_nn = (nnmodel.predict(X_train) > 0.5).astype("int32")
    NN_accuracy = accuracy_score(y_train,y_pred_nn)
    print("NEURAL NETWORK MODEL ACCURACY OVER TRAINING DATA : ",NN_accuracy)
    model_yaml = nnmodel.to_yaml()
    with open("../Models/NN_model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    nnmodel.save_weights("../Models/NN_model.h5")
    print("NEURAL NETWORK MODEL SAVED IN MODELS FOLDER.")


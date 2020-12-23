from keras.models import model_from_yaml
import numpy as np
import pickle

def load_logistic(input_features):
    loaded_model = pickle.load(open("../Models/logistic_regression.sav", 'rb'))
    print("\nLOGISTIC REGRESSION MODEL LOADED.\n")
    y_pred_logistic = loaded_model.predict(input_features)    
    return y_pred_logistic

def load_SVM(input_features):
    loaded_model = pickle.load(open("../Models/SVM.sav", 'rb'))
    print("\nSVM MODEL LOADED.\n")
    y_pred_SVM = loaded_model.predict(input_features)    
    return y_pred_SVM

def load_kNN(input_features):
    loaded_model = pickle.load(open("../Models/kNN.sav", 'rb'))
    print("\nkNN MODEL LOADED.\n")
    y_pred_kNN = loaded_model.predict(input_features)    
    return y_pred_kNN

def load_ridge(input_features):
    loaded_model = pickle.load(open("../Models/ridge.sav", 'rb'))
    print("\nRIDGE CLASSIFIER MODEL LOADED.\n")
    y_pred_ridge = loaded_model.predict(input_features)    
    return y_pred_ridge

def load_neural(input_features):
    input_features = np.asarray(input_features).astype('float32')
    yaml_file = open('../Models/NN_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights("../Models/NN_model.h5")
    print("\nNEURAL NETWORK MODEL LOADED.\n")
    y_pred_nn = loaded_model.predict_classes(input_features)    
    return y_pred_nn[0]
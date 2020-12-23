import os
import support_Scripts.utilities as utilities
import argparse
from support_Scripts.classifier_models import *
import support_Scripts.pre_process as pre_process
import warnings

warnings.filterwarnings("ignore")


DATA_DIR = "../dly532.csv"
models_path = "../Models"
try:
    os.mkdir(models_path)
except OSError:
    print ("Directory %s already present." % models_path)
else:
    print ("Directory %s created." % models_path)
    
parser = argparse.ArgumentParser(description="Fetch latest data and options for train models.")

parser.add_argument("-n","--newdata", help="Check for latest data online", type = str, metavar='',choices=["yes","no"],default = "no")
parser.add_argument("-m", "--model", metavar='',help="Model to be trained.", type = str, choices=["all","logistic","kNN","SVM","ridge","neural"], default = "logistic")

args = parser.parse_args()

if args.newdata=="yes":
    utilities.checkLatestVersion(DATA_DIR)
elif args.newdata=="no":
    if os.path.isfile(DATA_DIR) == False:
            utilities.checkLatestVersion(DATA_DIR)
    else:
        print ("Using existing dly532.csv file.")

# args = parser.parse_args()
if args.model=="all":
    print("\n\nTRAINING ALL AVAILABLE MODELS.\n")
    X_train, y_train = pre_process.preprocess_data(DATA_DIR)
    logistic_regression_model(X_train,y_train)
    SVM_model(X_train,y_train)
    kNN_model(X_train,y_train)
    ridgeModel(X_train,y_train)
    neuralNetwork_model(X_train,y_train)

elif args.model=="logistic":
    X_train, y_train = pre_process.preprocess_data(DATA_DIR)
    logistic_regression_model(X_train,y_train)

elif args.model=="SVM":
    X_train, y_train = pre_process.preprocess_data(DATA_DIR)
    SVM_model(X_train,y_train)
    
elif args.model=="kNN":
    X_train, y_train = pre_process.preprocess_data(DATA_DIR)
    kNN_model(X_train,y_train)

elif args.model=="ridge":
    X_train, y_train = pre_process.preprocess_data(DATA_DIR)
    ridgeModel(X_train,y_train)

elif args.model=="neural":
    X_train, y_train = pre_process.preprocess_data(DATA_DIR)
    neuralNetwork_model(X_train,y_train)


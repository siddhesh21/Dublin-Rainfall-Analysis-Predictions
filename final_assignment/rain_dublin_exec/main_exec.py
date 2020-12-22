import os
import utilities
import argparse
from classifier_models import *
import warnings

warnings.filterwarnings("ignore")


DATA_DIR = "./dly532.csv"
models_path = "./Models"
try:
    os.mkdir(models_path)
except OSError:
    print ("Directory %s already present." % models_path)
else:
    print ("Directory %s created." % models_path)
    
parser = argparse.ArgumentParser(description="Fetch latest data and options for train models.")

parser.add_argument("-d","--newData", help="Check for latest data online", type = str, metavar='',choices=["yes","no"],required=True,default = "no")
parser.add_argument("-m", "--model", metavar='',help="Model to be trained.", type = str, choices=["all","logistic","kNN","SVM","ridge","neural"],required=True, default = "all")

args = parser.parse_args()

if args.newData=="yes":
    utilities.checkLatestVersion(DATA_DIR)
elif args.newData=="no":
    if os.path.isfile("./dly532.csv") == False:
            utilities.checkLatestVersion(DATA_DIR)
    else:
        print ("Using existing dly532.csvm file.")

# args = parser.parse_args()
if args.model=="all":
    print("\n\nTRAINING ALL AVAILABLE MODELS.\n")
    X_train, y_train = pre_process.preprocess_data("./dly532.csv")
    logistic_regression_model(X_train,y_train)
    SVM_model(X_train,y_train)
    kNN_model(X_train,y_train)
    ridgeModel(X_train,y_train)
    neuralNetwork_model(X_train,y_train)

elif args.model=="logistic":
    X_train, y_train = pre_process.preprocess_data("./dly532.csv")
    logistic_regression_model(X_train,y_train)

elif args.model=="SVM":
    X_train, y_train = pre_process.preprocess_data("./dly532.csv")
    SVM_model(X_train,y_train)
    
elif args.model=="kNN":
    X_train, y_train = pre_process.preprocess_data("./dly532.csv")
    kNN_model(X_train,y_train)

elif args.model=="ridge":
    X_train, y_train = pre_process.preprocess_data("./dly532.csv")
    ridgeModel(X_train,y_train)

elif args.model=="neural":
    X_train, y_train = pre_process.preprocess_data("./dly532.csv")
    neuralNetwork_model(X_train,y_train)


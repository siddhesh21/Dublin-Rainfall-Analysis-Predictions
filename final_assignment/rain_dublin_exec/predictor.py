import os
import argparse
import warnings
import support_Scripts.pre_process as pre_process
import support_Scripts.model_load as model_load

warnings.filterwarnings("ignore")


DATA_DIR = "../dly532.csv"
parser = argparse.ArgumentParser(description="Choose Model for prediction out of the available options.")

parser.add_argument("-l", "--load", metavar='',help="Model to be trained.", type = str, choices=["logistic","kNN","SVM","ridge","neural"], default = "logistic")

args = parser.parse_args()

resultant = 0
input_features,y = pre_process.preprocess_data(DATA_DIR, date=True)

if args.load=="logistic":
    resultant = model_load.load_logistic(input_features)

elif args.load=="SVM":
    resultant = model_load.load_SVM(input_features)
    
elif args.load=="kNN":
    resultant = model_load.load_kNN(input_features)

elif args.load=="ridge":
    resultant = model_load.load_ridge(input_features)

elif args.load=="neural":
    resultant = model_load.load_neural(input_features)

if resultant == 1:
    print("\nAccording to prediction, it will rain on the specified date.")
else:
    print("\nAccording to prediction, it will NOT rain on the specified date.")
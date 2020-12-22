import numpy as np
import pandas as pd
import utilities

def preprocess_data(filename):
    names = ["DATE", "INDICATOR0","MAX_AIR_TEMP",
           "INDICATOR1","MIN_AIR_TEMP","I_GRASS_MIN_TEMP",
           "GRASS_MIN_TEMP","INDICATOR2","RAIN_MM",
           "MEAN_CBL_PRESSURE","MEAN_WINDSPEED_KNOT","INDICATOR3",
           "HIGHEST_10MIN_WINDSPEED","INDICATOR4","WIND_DIR_DEGREE",
           "INDICATOR5","HIGHEST_GUST","SUN_DURATION","DOS",
           "GLOBAL_RADIATION","SOIL_TEMP","POTENTIAL_EVAPOTRANSPIRATION",
           "EVAPORATION","SMD_WELL_DRAINED","SMD_MODERATELY_DRAINED","SMD_POORLY_DRAINED"]
    read_data = pd.read_csv(filename, names = names)

    read_data = pd.read_csv("dly532.csv", names = names)
    read_data = read_data.replace(r'^\s*$', np.nan, regex=True)
    read_data = read_data[read_data["SMD_POORLY_DRAINED"].notna()]
    read_data['HIGHEST_10MIN_WINDSPEED'] = read_data['HIGHEST_10MIN_WINDSPEED'].astype(float)
    read_data['WIND_DIR_DEGREE'] = read_data['WIND_DIR_DEGREE'].astype(float)
    read_data['HIGHEST_GUST'] = read_data['HIGHEST_GUST'].astype(float)
    read_data['GRASS_MIN_TEMP'] = read_data['GRASS_MIN_TEMP'].astype(float)
    read_data['EVAPORATION'] = read_data['EVAPORATION'].astype(float)
    read_data['DOS'] = read_data['DOS'].astype(float)
    read_data['GLOBAL_RADIATION'] = read_data['GLOBAL_RADIATION'].astype(float)
    read_data['SOIL_TEMP'] = read_data['SOIL_TEMP'].astype(float)
    read_data['SMD_WELL_DRAINED'] = read_data['SMD_WELL_DRAINED'].astype(float)
    read_data['SMD_MODERATELY_DRAINED'] = read_data['SMD_MODERATELY_DRAINED'].astype(float)
    read_data['SMD_POORLY_DRAINED'] = read_data['SMD_POORLY_DRAINED'].astype(float)
    
    read_data = read_data.drop(columns = ["I_GRASS_MIN_TEMP","GRASS_MIN_TEMP",
                                      "INDICATOR2","GLOBAL_RADIATION","DOS",
                                      "SOIL_TEMP","INDICATOR0","INDICATOR1",
                                      "INDICATOR2","INDICATOR3","INDICATOR4",
                                      "INDICATOR5","MEAN_CBL_PRESSURE",
                                      "SMD_WELL_DRAINED","SMD_MODERATELY_DRAINED",
                                      "SMD_POORLY_DRAINED"])

    read_data = read_data.fillna(read_data.mean())
    read_data.isnull().any()

    read_data.loc[read_data["RAIN_MM"] == 0, "RAIN_BOOLEAN"] = 0
    read_data.loc[read_data["RAIN_MM"] > 0, "RAIN_BOOLEAN"] =  1

    target_Arr = np.array([int(i) for i in read_data["RAIN_BOOLEAN"]])
    # target_Arr = target_Arr.pop(-1)

    read_data = read_data.drop(columns = ["RAIN_MM","DATE","RAIN_BOOLEAN"])

    read_data=utilities.Min_Max_Normalization(read_data)

    # Saving the latest date data separately for purpose of prediction.
    # read_data_presentDate = read_data.tail(1)

    # Removing the latest date data from the dataset.
    # read_data.drop(read_data.tail(1).index, inplace = True)

    read_data = read_data.drop(columns = ["MIN_AIR_TEMP","MAX_AIR_TEMP","POTENTIAL_EVAPOTRANSPIRATION"])

    norm_training_data = read_data.values

    return(norm_training_data, target_Arr)
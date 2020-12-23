import os
import pandas as pd

# Min-max normalization
def Min_Max_Normalization(dataframe,date=False):
    if date == False:
        dataframe=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
        return dataframe
    else:
        dates_data = dataframe[dataframe.columns[0]]
        dates = pd.DataFrame(data=dates_data,columns=['DATE'])
        columnNames = dataframe.columns
        columnNames = columnNames[1:]
        norm_frame=(dataframe[columnNames]-dataframe[columnNames].min())/(dataframe[columnNames].max()-dataframe[columnNames].min())
        dataframe = pd.concat([dates,norm_frame],axis=1,sort=False)
        return dataframe

def checkLatestVersion(pathname):
    if(os.path.exists(pathname)):
        os.remove(pathname)
        print("Deleted previous CSV file.")
        print("Downloading latest CSV file.")
        os.system("wget --no-check-certificate https://cli.fusio.net/cli/climate_data/webdata/dly532.csv")
        os.system("mv dly532.csv ..")
        print("Downloaded dataset.")
    else:
        print("Downloading dataset.")
        os.system("wget --no-check-certificate https://cli.fusio.net/cli/climate_data/webdata/dly532.csv")
        os.system("mv dly532.csv ..")
        print("Downloaded dataset.")

    with open(pathname, "r") as reading:
        data = reading.read().splitlines(True)
    reading.close()
    os.remove(pathname)
    with open(pathname, "w") as writing:
        writing.writelines(data[26:])
    writing.close()
    return None
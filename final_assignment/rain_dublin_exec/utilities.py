import os

# Min-max normalization
def Min_Max_Normalization(dataframe):
    dataframe=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
    return dataframe

def checkLatestVersion(pathname):
    if(os.path.exists(pathname)):
        os.remove(pathname)
        print("Deleted previous CSV file.")
        print("Downloading latest CSV file.")
        os.system("wget https://cli.fusio.net/cli/climate_data/webdata/dly532.csv")
        print("Downloaded dataset.")
    else:
        print("Downloading dataset.")
        os.system("wget https://cli.fusio.net/cli/climate_data/webdata/dly532.csv")
        print("Downloaded dataset.")

    with open(pathname, "r") as reading:
        data = reading.read().splitlines(True)
    reading.close()
    os.remove(pathname)
    with open(pathname, "w") as writing:
        writing.writelines(data[26:])
    writing.close()
    return None
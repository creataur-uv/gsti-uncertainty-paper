## Raw Data Reader


import os
import pandas


def csv(path=""):
    """
    Reads all the CSV files in a directory with same structure and returns a pandas dataframe collecting all the data distributed over different files.
    """
    entries = os.listdir(path=path)

    csv_files = list()

    for entry in entries:
        if entry.endswith(".csv"):
            csv_files.append(os.path.join(path, entry))

    df = list()

    for file in csv_files:
        data = pandas.read_csv(
            file, encoding="iso-8859-2", low_memory=False, sep=r",", header=0
        )
        df.append(data)

    return pandas.concat(df)

if __name__ == "__main__":
    print(len(csv("./data")))
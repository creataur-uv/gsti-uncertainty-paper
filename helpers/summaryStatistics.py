import scipy.stats as stats
import math

def summary_Statistics(x):
    """
    Function to output summary statistics of a linear array of numbers

    Parameters
    ------
    x : a linear array

    Returns
    dictionary with key-value pairs as follows
        n       number of observations
        mean    average of observations
        std     standard deviation
        skew    skewness
        kurt    kurtosis
        min     minimum value
        Q1      first quantile
        median  median value
        Q3      third quantile
        max     maximum value
    """
    res: dict[str, float] = {}
    description = stats.describe(x)
    res["n"] = description.nobs
    res["mean"] = description.mean
    res["std"] = math.sqrt(description.variance)
    res["min"] = description.minmax[0]
    res["max"] = description.minmax[-1]
    res["skew"] = description.skewness
    res["kurt"] = description.kurtosis
    return res


## For testing only
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis
    from scipy.stats import skew

    data = np.random.normal(loc=50, scale=10, size=1000000)
    result = summary_Statistics(data)
    for key, value in result.items():
        print(key, "\t\t", value)

    print(stats.describe(data))
    print(stats.describe(stats.trimboth(data, 0.2)))

    print(kurtosis(data), skew(data))
    plt.hist(data, bins=100, color="skyblue", edgecolor="black")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()

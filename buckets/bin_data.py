import numpy as np

def standardarray(x):
    """Make sure an array is a np.array object.

    Parameters
    ----------
    x : array-like
        Array-like object to turn into a numpy array.

    Returns
    -------
    x : np.array
        Converted array.

    """
    if type(x) != np.ndarray:
        x = np.array(x)
    return x

def bin_x(x, y, gx):
    """Bins the input data `x` into centers at `gx` and calculates the mean of `y` in those bins.

    Parameters
    ----------
    x : array-like
        `x` value to bin `y` value data by.
    y : array-like
        `y` value to average over each `x` bin.
    gx : array-like
        Bin centers over which to calculate averages of `y`.

    Returns
    -------
    bin_means : np.array
        Mean of y in each bin.
    bin_std : np.array
        Standard deviation of y in each bin.
    N : np.array
        Number of elements in each bin.
    """
    # Make sure the format is correct.
    x = standardarray(x)
    y = standardarray(y)
    gx = standardarray(gx)

    # Remove NaNs here.
    good_x = np.where(~np.isnan(x))[0]
    good_y = np.where(~np.isnan(y))[0]
    good_idx = np.intersect1d(good_x, good_y)
    x = x[good_idx]
    y = y[good_idx]

    # Deal with edge factors.
    binwidth = np.diff(gx)
    gx = np.hstack((gx[0] - binwidth[0]/2., gx[:-1] + binwidth/2., gx[-1] + binwidth[-1]/2))

    # Determine which bin each y value belongs to.
    binind = np.digitize(x, gx)
    # Take the sum of each bin and then divide it by the total count in each bin (average).
    bin_means = [y[binind == i].mean() for i in range(1, len(gx))]
    bin_std = [y[binind == i].std() for i in range(1, len(gx))]
    N = [len(y[binind == i]) for i in range(1, len(gx))]

    return bin_means, bin_std, N

def bin_n(x, y, n):
    """Bins the input data `x` into 'n' bins and calculates the mean of `y` in those bins. Each bin contains the same number of elements.

    Parameters
    ----------
    x : array-like
        `x` value to bin `y` value data by.
    y : array-like
        `y` value to average over each `x` bin.
    n : int
        Number of elements to include in each bin.

    Returns
    -------
    X : np.array
        Bin centers.
    Y : np.array
        Average values of `y` in each bin.
    S : np.array
        Standard deviation of `y` in each bin.
    Sm : np.array
        Standard error of `y` in each bin.
    """
    # Make sure the format is correct.
    x = standardarray(x)
    y = standardarray(y)

    # Sort the x values given.
    ind = np.argsort(x)
    x = x[ind]
    # Sort the y values according to this x-sorting.
    y = y[ind]

    # Create the bins.
    min_i = np.arange(0,len(x),n)
    max_i = np.arange(0,len(x),n) - 1
    min_i = min_i[:-1]
    max_i = max_i[1:]
    max_i[-1] = len(x)

    # Initialize the vectors holding the binned values.
    X = np.ones(min_i.shape)
    Y = np.ones(min_i.shape)
    S = np.ones(min_i.shape)
    Sm = np.ones(min_i.shape)

    # Calculate the average within each bin and put into the initialized vectors.
    for i in range(len(max_i)):
        X[i] = np.mean(x[min_i[i]:max_i[i]])
        Y[i] = np.mean(y[min_i[i]:max_i[i]])
        S[i] = np.std(y[min_i[i]:max_i[i]])
        Sm[i] = np.std(y[min_i[i]:max_i[i]])/np.sqrt(len(y[min_i[i]:max_i[i]]))

    return X, Y, S, Sm

buckets
=======

Some better binning schemes for data visualization.

## Usage

To install, clone the library into a local repo, then run `pip install .` within the directory.

`bin_n` allows for binning with an equal number of elements in each bin. To use:

```
from buckets import bin_n

X, Y, S, Sm = bin_n(x, y, n)

```

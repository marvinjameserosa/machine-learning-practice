import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import k_means

# Data processing
rawData = load_digits()
print(rawData)
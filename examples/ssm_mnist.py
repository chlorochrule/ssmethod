# -*- coding: utf-8 -*-
"""
An example that demonstrate the classification using subspace method.

The script classifies digits of mnist database.
"""

import numpy as np
import time
from sklearn.datasets import fetch_openml, fetch_mldata
from sklearn.model_selection import train_test_split

from ssmethod import SubspaceClassifier


# mnist = fetch_mldata('mnist original', data_home='.')
mnist = fetch_openml('mnist_784', data_home='.', version=1)
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)
for i in [10, 20, 40, 60, 80, 100, 200, 300]:
    start = time.time()
    ssc = SubspaceClassifier(n_components=i)
    ssc.fit(X_train, y_train)
    pred = ssc.predict(X_test, y_test)
    print('n_components: {}'.format(i))
    print(ssc.accuracy_)
    print('computational time: {}'.format(time.time() - start))

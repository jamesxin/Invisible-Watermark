#!/usr/bin/env python
# encoding: utf-8

from sklearn.cluster import DBSCAN
import numpy as np
data = np.random.rand(500,3)

db = DBSCAN(eps=0.12, min_samples=1).fit(data)
labels = db.labels_

from collections import Counter

print Counter(labels)


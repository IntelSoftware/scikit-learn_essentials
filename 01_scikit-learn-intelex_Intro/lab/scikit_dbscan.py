
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
from sklearn.cluster import DBSCAN
from daal4py.oneapi import sycl_context

X = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
with sycl_context("gpu"):
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print("DBSCAN components: ", clustering.components_, "\nDBSCAN labels: ",clustering.labels_)

resultsDict = {}
resultsDict['X'] = X
resultsDict['labels'] = clustering.labels_
resultsDict['components'] = clustering.components_
import pickle
with open('resultsDict.pkl', 'wb') as handle:
    pickle.dump(resultsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

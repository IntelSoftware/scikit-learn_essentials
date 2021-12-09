import warnings
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
import dpctl
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_dictionary(fn):
    import pickle
    # Load data (deserialize)
    with open(fn, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

XenoSupermanGalaxy = read_dictionary('XenoSupermanGalaxy.pkl')
GFFA = read_dictionary('GFFA.pkl')

plt.style.use('dark_background')

# dataset is subset of stars from each galaxy
TrainingSize = min(len(GFFA['Stars']), len(XenoSupermanGalaxy['Stars'] ) ) 

collision = dict()
collision['Arms'] = np.vstack((GFFA['Arms'].copy(), XenoSupermanGalaxy['Arms'].copy()))
collision['CenterGlob'] = np.vstack((GFFA['CenterGlob'].copy(), XenoSupermanGalaxy['CenterGlob'].copy()))
collision['Stars'] = np.vstack((GFFA['Stars'].copy(), XenoSupermanGalaxy['Stars'].copy()))
collision['Stars'].shape

# get the index of the stars to use from XenoSupermanGalaxy
XenoIndex = np.random.choice(len(XenoSupermanGalaxy['Stars']), TrainingSize, replace=False)
# get the index of the stars to use from GFFAIndex
GFFAIndex = np.random.choice(len(GFFA['Stars']), TrainingSize, replace=False)

# create a list with a labelforeahc item in the combined training set
# the first hald of the list indicates that class 0 will be for GFFA, 1 will be XenoSupermanGalaxy
y = [0]*TrainingSize + [1]*TrainingSize
# Stack the stars subset in same order as the labels, GFFA first, XenoSupermanGalaxy second
trainGalaxy = np.vstack((GFFA['Stars'][GFFAIndex], XenoSupermanGalaxy['Stars'][XenoIndex]))  

x_train, x_test, y_train, y_test = train_test_split(trainGalaxy, np.array(y), train_size=0.05)

K = 3
myModels = {'KNeighborsClassifier':KNeighborsClassifier(n_neighbors = K) , 
            'RandomForestClassifier': RandomForestClassifier(n_jobs=2, random_state=0), 
            #'SVC':SVC(gamma='auto'), 
           }
TrainingSize = [.001, .01, .05, .1, .8]
bestScore = {}
hi = 0
K = 3

for d in dpctl.get_devices():
    if d.is_gpu:
        device = dpctl.select_gpu_device()
    else:
        device = dpctl.select_cpu_device() 
            
for tsz in TrainingSize:
    x_train, x_test, y_train, y_test = train_test_split( \
                trainGalaxy, np.array(y), train_size=tsz)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    for name, modelFunc in myModels.items():       
        start = time.time()
        model = modelFunc
        print(device.device_type)
        with dpctl.device_context(device):
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
        print('Display results of {} classification'.format(name))
        print('  K: ',K)
        print('  Training size: ', tsz)
        print('  y_train.shape: ',y_train.shape)
        roc = roc_auc_score(y_test, y_pred)
        print('  roc_auc_score: {:4.1f}'.format(100*roc))
        print('  Time: {:5.1f} sec\n'.format( time.time() - start))
        if roc > hi:
            hi = roc
            bestScore = {'name': name,
                    'roc':roc, 
                    'trainingSize':tsz, 
                    'confusionMatrix': confusion_matrix(y_test, y_pred), 
                    'precision': 100*precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary') }
print('bestScore: name', bestScore['name'])
print('bestScore: confusion Matrix', bestScore['confusionMatrix'])
print('bestScore: precision', bestScore['precision'])
print('bestScore: recall', bestScore['recall'])
print('bestScore: roc', bestScore['roc'])

# Notices & Disclaimers

# Intel technologies may require enabled hardware, software or service activation.

# No product or component can be absolutely secure.
# Your costs and results may vary.

# © Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. 
# *Other names and brands may be claimed as the property of others.

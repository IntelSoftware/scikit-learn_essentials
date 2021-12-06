# daal4py Kmeans example for shared memory systems
import pickle

from sklearnex import patch_sklearn
patch_sklearn()

from daal4py.oneapi import sycl_context
from daal4py.oneapi import sycl_buffer
from sklearn.cluster import KMeans
import numpy as np
import os

def write_results(resultsDict):
    print("write_results...")
    file_to_write = open("resultsDict.pkl", "wb")
    pickle.dump(resultsDict, file_to_write)
    file_to_write.close()
    print("write complete...")
    
# let's try to use pandas' fast csv reader
try:
    import pandas
    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)

# Commone code for both CPU and GPU computations
def compute(data, nClusters, maxIter, method):    
    kmeans = KMeans(nClusters, init='random', max_iter=maxIter, random_state=0)
    #kmeans = KMeans(nClusters, random_state=0, init='random', maxIter=5)
    y_km = kmeans.fit(data)
    pred_y = kmeans.fit_predict(data)
    
    print("kmeans.labels_")
    print(kmeans.labels_)   
    
    print("kmeans.cluster_centers_")
    #print(kmeans.cluster_centers_)    
    print("\nFirst 3 cluster centers:\n", kmeans.cluster_centers_[0:3])
    resultsDict = {}
    resultsDict['y_km'] = y_km
    resultsDict['pred_y'] = pred_y
    resultsDict['kmeans.labels_'] = kmeans.labels_
    resultsDict['kmeans.cluster_centers_'] = kmeans.cluster_centers_
    return resultsDict


# At this moment with sycl we are working only with numpy arrays
def to_numpy(data):
    try:
        from pandas import DataFrame
        if isinstance(data, DataFrame):
            return np.ascontiguousarray(data.values)
    except ImportError:
        pass
    try:
        from scipy.sparse import csr_matrix
        if isinstance(data, csr_matrix):
            return data.toarray()
    except ImportError:
        pass
    return data


def main(readcsv=read_csv, method='randomDense'):
    infile = os.path.join('data', 'batch', 'kmeans_dense.csv')
    nClusters = 20
    maxIter = 5
    
    print("output expected below:")
    
    # Load the data
    data = readcsv(infile, range(20), t=np.float32)   

    # convert to numpy
    data = to_numpy(data)

    # determine if GPU available:
    dpctl_available = False
    try:
        # modern approach for SYCL context is to use dpctl & sklearnex._config supported in 
        # Intel(R) Extension for Scikit-learn* 2021.4 
        import dpctl
        from sklearnex._config import config_context
        dpctl_available = True
    except ImportError:
        try:
            # older approach for SYCL context is to use dall4py  
            from daal4py.oneapi import sycl_context
            print("*" * 80)
            print("\ndpctl package not found, switched to daal4py package\n")
            print("*" * 80)
        except ImportError:
            print("\nRequired packages not found, aborting...\n")
            exit()

    gpu_available = False
    if dpctl_available:
        try:
            with config_context(target_offload='gpu'):
                gpu_available = True
                def get_context(device):
                    return config_context(target_offload=device)
        except Exception:
            gpu_available = False
    else:
        try:
            with sycl_context('gpu'):
                gpu_available = True
                def get_context(device):
                     return sycl_context(device)
        except Exception:
            gpu_available = False
        
    # It is possible to specify to make the computations on GPU
    if gpu_available:
        with get_context('gpu'):
            print('Running on GPU: ')    
            resultsDict = compute(data, nClusters, maxIter, method) 

    write_results(resultsDict)
if __name__ == "__main__":
    result = main()    
    print('All looks good!')

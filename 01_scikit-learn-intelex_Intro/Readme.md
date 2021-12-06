## Title
 Introduction to Intel(R) Extension for Scikit-learn: This is module 1 of the AI essentials, Intel(R) Extension for Scikit-learn training series
  
## Requirements
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel Devcloud
  
## Purpose
The initial hands-on exercises in this notebook introduce you to Intel(R) Extension for Scikit-learn. Also, it familiarizes you with the use of Jupyter notebooks as a front-end for all training exercises. This workshop is designed to be used on the Devcloud and includes details on submitting batch jobs on the Devcloud environment.

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Install Directions

The Jupyter notebooks are tested and can be run on Intel Devcloud.
Below are the steps to access these Jupyter notebooks on Intel Devcloud
1. Register on [Intel Devcloud](https://intelsoftwaresites.secure.force.com/Devcloud/oneapi)
2. Go to the "Terminal" in the Intel Devcloud
3. Type in the below command to download the oneAPI-essentials series notebooks into your Devcloud account
    /data/oneapi_workshop/get_jupyter_notebooks.sh
4. Navigate to AI_Essentials folder and open the Welcome.ipynb, click on "Module 1 - Intel(R) Extension for Scikit-learn" notebook and follow the instructions

## dependencies:
 - sklearn
 - sklearnex
 - daal4py
 - matplotlib
 - pandas
 - numpy
 - time
 - timeit
 
The script demonstrates a very simple way to leverage the following sklearn algorithms (optimized within Intel(r) Extensions for scikit-learn)
        k_means_init_x,
        k_means_random,
        linear_regression,
        logistic_regression_lbfgs,
        logistic_regression_newton,
        dbscan
        
The algorithms are run on a gpu on the Intel DevCloud or Aurora server. a couple of these cells demonstrate the advantage of sklearn-ex over the stock sklearn.

to run, choose one fo the following lines to execute as a python script (embedded in a bash script)
`. run_sklearn_sycl.sh`
`. run_scikit_dbscan.sh`
`. run_daal4py.sh`
`. run_scikit_kmeans.sh`

equivalently - you can pull the python script out of the bash script and run as follows:
`python lab/scikit_dbscan.py`
and so on for the other modules

## datasets
the data are very small, very simple matrices to demonstrate the methods only:
X = np.array([[1,  2], [1,  4], [1,  0],
              [10, 2], [10, 4], [10, 0]])



## Title
 Introduction to Intel(R) Extension for Scikit-learn: This is module 2 of the AI essentials, Intel(R) Extension for Scikit-learn training series
  
## Requirements
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel Devcloud
  
## Purpose
The initial hands-on exercises in this notebook introduce you to Intel(R) Extension for Scikit-learn. Also, it familiarizes you with the use of Kmeans using Intel Extension for Scilit-learn

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

Requirements:
 - sklearn
 - numpy
 - pickle
 - pandas
 - sklearnex
 - daal4py
 - matplotlib (for optional visualization output)

Everything else is core python.

The data is a dense matrix of data, found in data/batch/kmeans_dense.csv, with dimensions: (9999, 20)
roughly distributed in a gausisian fashion single mode per column, with extreme values near -200, +200, centered on zero.

We will compute several sklearn functions within a SYCL device context, leveraging gpu compute via the following code snippet:
        import dpctl
        from sklearnex._config import config_context
        device = 'gpu'
        def get_context(device):
            if dpctl_available:
                return config_context(target_offload=device)
            return sycl_context(device)
        
        with  get_context('gpu'):           
            pca = PCA(n_components=n_components)
            PCA_fit_transform = pca.fit_transform(NP_images_STD) 
            k_means = KMeans(n_clusters = knee, init='random')
            db = DBSCAN(eps=EPS, min_samples = n_samples).fit(PCA_fit_transform)
            km = k_means.fit(PCA_fit_transform)
            
To run the python code, run the wrapping bash shell script such as :
run `. run_clustering_streamlined.sh`
run `. run_kmeans_kernel2.sh`

these scripts simply wrap the following:
`python lab/kmeans_gpu.py`

For questions, please contact Bob Chesebrough: bob.chesebrough@intel.com




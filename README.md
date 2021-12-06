# oneAPI training Jupyter notebooks 

The purpose of this repo is to be the central aggregation, curation, and distribution point for Juypter notebooks that are developed in support of oneAPI training programs (e.g., oneAPI Essentials Series). 

The Jupyter notebooks are tested and can be run on Intel Devcloud.
Below are the steps to access these Jupyter notebooks on Intel Devcloud
1. Register on [Intel Devcloud](https://intelsoftwaresites.secure.force.com/Devcloud/oneapi)
2. Go to the "Terminal" in the Intel Devcloud
3. Type in the below command to download the oneAPI-essentials series notebooks into your Devcloud account
    /data/oneapi_workshop/get_jupyter_notebooks.sh
    
## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# The organization of the Jupyter notebook directories is a follows:

| Notebook Name | Owner | Description |
| --- | --- | --- |
|[oneAPI_Intro](01_scikit-learn-intelex_Intro/01_scikit-learn-intelex_Intro.ipynb)|Bob.Chesebrough@intel.com| + Introduction and Motivation for using sklearn algorithms which have have been optimzied in the Intel(r) Extensions for scikit-learn* or its subordinate library, daal4py..<br> + Explore simple approaches for invoking SYCL context against a multitude of sklearn algorithsm: <br> +  + k_means_init_x<br> +  + k_means_random<br> +  + logistic_regression_lbfgs<br> +  + logistic_regression_newton<br> +  + dbscan |
| --- | --- | --- |
|[sklearn-ex Kmeans](02_scikit-learn-intelex_Kmeans.ipynb)|Bob.Chesebrough@intel.com| + Use Data parallel Control (__dpCtl__) to manage different devices<br> + Use __sklearn-ex__ and __daal4py__ libraries<br> + Explore Kmeans with differing contexts including __cpu, gpu and distributed__ |
| --- | --- | --- |
|[Image Clustering](03_scikit-learn-intelex_Image_Cluster/03_ImageClustering.ipynb)|Bob.Chesebrough@intel.com| Use multiple algorthms: <br> +  PCA, <br> +  kmeans, <br> +  DBSCAN <br>all within a given SYCL device context to perform image clustering of a batch of images |
| --- | --- | --- |
|[Classifcation of galactic stars using kNN/KDTree](04_scikit-learn-intelex-KNN-using_kdtree/04_AnalyzeGalaxyBatch.ipynb)|Bob.Chesebrough@intel.com| + What is Sub-Goups and Motivation<br>+ Quering for __sub-group info__<br>+ Sub-group __collectives__<br>+ Sub-group __shuffle operations__ |




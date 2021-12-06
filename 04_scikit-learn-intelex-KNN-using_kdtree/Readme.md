## Title
 04_analyzeGalaxyBatch: This is part 4 of the oneAPI essentials training series
  
## Requirements
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel Devcloud
  
## Purpose
The hands-on exercises in this notebook show how to implement a SYCL gpu context on multiple classification algorithms in sklearn using oneAPI and the Intel(R) Extensions for scikit-learn

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
4. Navigate to oneAPI_Essentials folder and open the Welcome.ipynb, click on "Module 3 - DPC++ Unified Shared Memory" notebook and follow the instructions

## Datasets
This is a small galaxy intersection application and uses a knn classifcation as well as KDtree for distance groupings less than 3 light years. The data was synthesized using 3D gaussian distributios for center of galaxy as well as small globular clusters in arms, the arms were sythesized accoring to equation mentioned in "A New Formula Describing the Scaffold Structure of Spiral Galaxies" https://arxiv.org/ftp/arxiv/papers/0908/0908.0892.pdf

Requirements:
 - dpctl
 - sklearnex._config
 - sklearn
 - numpy
 - pickle
 - matplotlib (for optional visualization output)

Everything else is core python.

There are 2 pickle files in this folder that are useful for this classifcaition app.  'XenoSupermanGalaxy.pkl' contains the synthesized 3D coordinates for ~80,000 stars and represents a ficticious galaxy, XenoSupermanGalaxy, which popular culture stipulates is the galaxy containing Supermans planet of Krypton

'GFFA.pkl' contains the synthesized 3D coordinates for ~80,000 stars and represents a ficticious galaxy, GFFA, which popular culture describes as the short hand for a "Galaxy Far, Far, Away", from Star Wars canon. These data are comnbined into a superset colliding galaxy.

## Running
This script first combine the two datasets into a superset of data. It will take a fractions  somethng like 1% to 10% of the data known from each galaxy as labels and use different classifcation algorithms optimized in Intel Extensions for Scikit-learn* (such as RandomForest, and KNN). Once classified we will use KDtree to find all stars within 3 lights from each other.

To run the script, use `python 04_analyzeGalaxyBatch.py`.

For questions, please contact Boib Chesebrough: bob.chesebtrough@intel.com

To run the workload on different devices, use modified `cluster_images_intel.py` script with support of Intel CPUs and GPUs.

### 04_AnalyzeGalaxyBatch.ipynb 
Contains same functionality as the python script. 

### 04_AnalyzeGalaxyScikit-learnClassificationIncludingKNNwKdtree.ipynb.ipynb
Contains same functionality as the python script plus code to synthesize different galaxies and explore the data more thoroughly.




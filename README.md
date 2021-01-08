<h1 align="center">Clustering-with-queries</h1>
 <p align="center"><b>An implementation of k-means clustering in a Semi-Supervised Clustering(SSAC) framework</b></p>

## Overview
This is an implementation of this [paper](https://arxiv.org/pdf/1606.02404.pdf). The algorithm is allowed to interact with a domain expert that can answer whether 2 instances belong to the same cluster. The cluster is found using O(k\*log n) queries and in O(n\*k\*log n) time with high probability. The domain expert is simulated by an oracle which loads an already labeled clustering and provides a query function. 

## Usage
### To run on example datasets 
- Git clone the repo  
- Setup the conda environment using the `environment.yml` file

        conda env create --file environment.yml

- Activate the environment 

        conda activate clustering

- Then simply run `clusterise.py`

        python clusterise.py
### To run on your own dataset
- Add your dataset in the required format in npz format in the `dataset` folder. Look at `gen.py` for details
- In `clusterise.py`, replace `data5.npz` with your filename in the Oracle object initialisation  
- Run `clusterise.py` as above 




## License

Licensed under the [MIT License](./LICENSE).
# Collaborative Variational Bandwidth Auto-encoder

 The codes are associated with the following paper:
 >**Collaborative Variational Bandwidth Auto-encoder for Recommender Systems,**  
 >Yaochen Zhu and Zhenzhong Chen. ArXiv Preprints. [[pdf]](https://arxiv.org/abs/2105.07597). 


## Environment

The codes are written in Python 3.6.5 with the following packages.  

- numpy == 1.16.3
- pandas == 0.21.0
- tensorflow-gpu == 1.15.0
- tensorflow-probability == 0.8.0

## Datasets

The processed datasets can be found [here](https://www.dropbox.com/s/wa0ia7svl7mnuvv/data.zip?dl=0). 

For usage, create a data folder and move in the unzipped datasets.

## Examples to run the codes

### To reproduce the comparison results in Table 2:

- **Layerwise pretrain the user feature VAE**: 

    ```python pretrain_vae.py --dataset Name --split [0-9]```
- **Iteratively train the collarabotive and feature part of VBAE**:

    ```python train_vbae.py --dataset Name --split [0-9]```
- **Evaluate the model and summarize the results into a pivot table**
    
    ```python predict.py --dataset Name --split [0-9]```
    
    ```python summarize.py```

### To reproduce the bandwidth analysis results in Table 3:
- **Summarize the average, std and corr of bandwidth into the model folder**

    ```python analyse_bandwidth.py --dataset Name --split [0-9]```


 For more advanced argument usage, run the code with --help argument.

## **Citation**

If you find our codes helpful, please kindly cite the following paper. Thanks!

	@article{vbae_zhu2021,
	  title={ollaborative Variational Bandwidth Auto-encoder for Recommender Systems},
	  author={Zhu, Yaochen and Chen, Zhenzhong},
	  booktitle={arXiv preprint arXiv:2105.07597},
	  year={2021},
	}	
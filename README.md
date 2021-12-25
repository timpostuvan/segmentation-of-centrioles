# CS-433 Machine Learning - Semantic Segmentation of Centrioles in Human Cells and Assigning Them to Nuclei
This repository contains the code for the second ML project 2 ML4Science, performed in the Gönczy Lab – Cell and Developmental Biology at EPFL.

## Team members
* Antoine Daeniker: antoine.daeniker@epfl.ch
* Oliver Becker: oliver.becker@epfl.ch
* Tim Poštuvan: tim.postuvan@epfl.ch

## Project Description
Determination of the number of centrioles is central to better understand their role in cancer since centrosome amplification often occurs in cancer cells. To detect centrioles, we propose an approach based on semantic segmentation of centrioles. Furthermore, we segment nuclei and assign centrioles to them in an unsupervised manner. The assignment is done by defining the problem as minimum weight matching in a bipartite graph with prior greedy matching. This allows us to incorporate domain knowledge that each nucleus can have at most four centrioles. Our approach is also evaluated at all stages, except for nuclei segmentation one. 

### Install requirements
Run : 
```
pip install -r requirements.txt
```

### Data
Download the data from this drive: https://drive.google.com/drive/folders/1pQUSt-qwXfVtIBig7JElVzEM0tWha2I0?usp=sharing. The folder structure should look like:
```
$root
|-- dataset
`-- experiments
|   
`-- src
```

### Create all-channel dataset
To merge the channel images into one, use `create_channel_images.ipynb` notebook.

## Run
First, download the data to get the tifs, run the U-Net to get predictions, run center detection to get centrioles's centers, and run the matching to assign centrioles to nuclei. 

### Run U-Net
To generate predictions of the U-Net on the test set, run the following command inside the `src` folder:
```
python3 run.py --config ../experiments/full_experiment_single_channel.json --num_workers 0
```
       
The script automatically generates predictions on the test set if there are trained weights in the folder `source/checkpoints`. To train the model from scratch, remove the weights. 

Explore how to test or train the U-Net with the corresponding json under `experiments`:

`full_experiment_single_channel`  - runs the U-Net on the single-channel images.

`full_experiment_all_channels` – runs the U-Net on the all-channel images.

Furthermore, you can run it with no attention (`no_attention_single_channel` and `no_attention_all_channels`) and no data augmentation (`no_data_augmentation_all_channels` and `no_data_augmentation_single_channel`).

### Run center detection
The notebook `pred_mask_view.ipynb` loads predicted masks and performs center detection. Then, it writes all detected coordinates in a csv file named `predictions_annotation.csv`, which is located in `dataset/single-channel-images` folder.
       
### Run matching
The notebook `matching.ipynb` shows and explains our matching procedure, everything from loading the tif image to using StarDist, and creating a matching. 
The notebook `matching_bipartite.ipynb` shows and explains bipartite only matching procedure. 

## Paper
The paper about the project is located in `paper` folder.

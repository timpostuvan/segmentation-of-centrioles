# CS-433 Machine Learning - Semantic Segmentation of Centrioles in Human Cells and Assigning Them to Nuclei
This repository contains the code for the second ML project 2 ML4Science, performed in the Gönczy Lab – Cell and Developmental Biology at EPFL.

## Team members
* Antoine Daeniker antoine.daeniker@epfl.ch
* Oliver Becker oliver.becker@epfl.ch
* Tim Poštuvan tim.postuvan@epfl.ch

## Project Layout
Determination of the number of centrioles is central to
better understand their role in cancer since centrosome amplification
often occurs in cancer cells. To detect centrioles, we propose approach
based on semantic segmentation of centrioles. Furthermore, we seg-
ment nuclei and assign centrioles to them in unsupervised manner.
Assignment is done by defining the problem as minimum weight
matching in bipartite graph with prior greedy matching. This allows
to incorporate domain-knowledge that each nuclei can have at most
four centrioles. Our approach is also evaluated at all stages, except for
nuclei segmentation.

## Run
### Download the data
Download the data in this drive: https://drive.google.com/drive/folders/1pQUSt-qwXfVtIBig7JElVzEM0tWha2I0?usp=sharing.

$root

|-- dataset

|-- experiments

`
|
`

|-- src


### Install requirement
run : [pip install -r requirements.txt]

### Run U-Net
To run the test of the U-Net and give out predictions use the following command inside the `src` folder:
        
       python 3 run.py --config ../experiments/full_experiment_single_channel.json --num_workers 0

Explore how to train the U-Net with the corresponding json under `experiments`:
`full_experiment_single_channel`  - Runs the U-Net on the single channel images.

`full_experiment_all_channels` – Runs the U-Net on the images which combine all channels.

Further you can run it with no attention for single and all channels (`no_attention_single_channel` and no_attention_all_channels) and no data augmentation (`no_data_augmentation_all_channels` and `no_data_augmentation_single_channel`)

### Run center detection
The notebook `perd_mask_view.ipynb` load every data according to our masks predictions and perform the center detection. Then it will write all detected coordinates in a csv file named `predictions_annotation.csv` which will be located in `dataset/single-channel-images` folder.
       
### Run Matching
The notebook `matching.ipynb` shows and explains our matching procedure, everything from loading the tif image to using StarDist and creating a matching. 
The notebook `matching_bipartite.ipynb` shows and explains bipartite only matching procedure. 

## Data and preparations

### Data
Because of the tif images and their size the data is too big for GitHub. The data can instead be found in: https://drive.google.com/drive/folders/1pQUSt-qwXfVtIBig7JElVzEM0tWha2I0?usp=sharing

### Create one image from all channels
To merge the channel images into one image we created the 'create_channel_images.ipynb' notebook. This goes through the procedure to create the all channel images.

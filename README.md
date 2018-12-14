# CSCI2470-Final-Project

Problem statement comes from Kaggle's [Human Protein Atlas Image Classification Competition](https://www.kaggle.com/c/human-protein-atlas-image-classification).

## Data

The input images are 512x512 PNG files. Each sample consists of 4 images (color channels) consisting of blue, green, red, and yellow

## Predictions

This is a multilabel classification problem with the following labels

0.  Nucleoplasm  
1.  Nuclear membrane   
2.  Nucleoli   
3.  Nucleoli fibrillar center   
4.  Nuclear speckles   
5.  Nuclear bodies   
6.  Endoplasmic reticulum   
7.  Golgi apparatus   
8.  Peroxisomes   
9.  Endosomes   
10.  Lysosomes   
11.  Intermediate filaments   
12.  Actin filaments   
13.  Focal adhesion sites   
14.  Microtubules   
15.  Microtubule ends   
16.  Cytokinetic bridge   
17.  Mitotic spindle   
18.  Microtubule organizing center   
19.  Centrosome   
20.  Lipid droplets   
21.  Plasma membrane   
22.  Cell junctions   
23.  Mitochondria   
24.  Aggresome   
25.  Cytosol   
26.  Cytoplasmic bodies   
27.  Rods & rings

## Evaluation

Evaluation is based on the [macro F1 score](https://en.wikipedia.org/wiki/F1_score).

### Benchmarks

| Method      | F1 Score | 
|-------------|----------|
| All Zeros   | 0.019    |
| All Classes | 0.111    |s

### Repository structure
You can find the Vanilla CNN and pretrained models under /project/ file. 

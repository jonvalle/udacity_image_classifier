# Udacity: Image Classifier

This is the final work for a course called: "AI with Python Nanodegree"

## In this project we will cover: 
* Python basics
* Python libraries: Numpy, Pandas, Matplotlib, Torch
* Jupyter Notebook
* Linear Algebra
* Machine learning: train models and pre-trained models
* Use local and remote gpu/cpu trainers
* Final project: classify an image based on a previously trained model. 

## Requirements
* Conda 4.5.9 
* Python 3.6.6 (incl in Conda)
* Numpy 1.14 (incl in Conda)
* Pandas 0.23 (incl in Conda)
* Pip 10.0.1 (incl in Conda)
* MatplotLib (incl in Conda)
* PILow (incl in Conda)
* Jupyter Notebook (incl in Conda)
* Pytorch 
** conda install pytorch-cpu -c pytorch
** pip3 install torchvision

## Jupyter Notebook
1. Clone this repo
2. Open bash. 
3. Run "jupyter notebook" on the root of this project
4. Navigate to file: Image Classifier Project.ipynb

## Command Line
### Using a trained model
1. Find predict.py file and run it
2. Use the provided parameters below, or change accordingly:
> python predict.py --top_k 8 --gpu --checkpoint checkpoint.pth --input flowers/valid/38/image_05819.jpg
(requires a checkpoint.pth file with the trained model. Contact me if you need one). 

### Training your own model
1. Find train.py file and run it
2. Use the provided parameters below, or change accordingly:
> python train.py --data_dir flowers --arch vgg13 --epochs 5 --gpu 
3. Find predict.py file and run it
4. Use the provided parameters below, or change accordingly:
> python predict.py --top_k 8 --gpu --checkpoint checkpoint.pth --input flowers/valid/38/image_05819.jpg

Happy coding!

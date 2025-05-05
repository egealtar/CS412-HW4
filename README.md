# CS412 - Machine Learning: Homework 4
## Transfer Learning for Gender Classification on the CelebA Dataset

### Overview
This project implements transfer learning techniques for binary gender classification using the CelebA dataset. A pretrained VGG-16 model is adapted and fine-tuned using different strategies and learning rates.

### Files
- `CS412-HW4-EgeAltarDeniz.ipynb`: Jupyter notebook containing all code and implementation
- `CS412-HW4-EgeAltarDeniz.pdf`: Comprehensive project report

### Methodology
The project explores two different fine-tuning strategies:
1. Freezing all convolutional layers and training only the classifier head
2. Freezing all weights, but fine-tuning the last convolutional block along with the classifier head

Each strategy is tested with two different learning rates (0.0001 and 0.00001), resulting in four different configurations.

### Results
The best performance was achieved by the second strategy (fine-tuning the last convolutional block + classifier head) with a learning rate of 0.0001, resulting in 97.93% validation accuracy and 97.83% test accuracy.

### Requirements
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- scikit-learn

# PyTorch CNN Radio Signal Classifier for RTL SDR

This repository contains a radio signal classifier using deep learning. The classifier is trained on a dataset of various radio signal types, such as WFM, TV, TETRA, GSM, and others. The classifier is built using PyTorch and tested using the RTL-SDR software-defined radio (SDR) device.

## Requirements

- Python 3.x
- PyTorch
- Numpy
- SciPy
- RTL-SDR Python library
- Dataset (in 'training_data' directory)

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/radio-signal-classifier.git
cd radio-signal-classifier
```

2. **Install the required packages:**

```bash
    pip install -r requirements.txt
```
Prepare the dataset in a directory named training_data. The dataset should have subdirectories for each class of radio signal. Make sure the dataset is compatible with the provided dataset2.py script.

3. **Run the script:**

```
    python predict_scan.py
```
This will train the convolutional neural network (CNN) model on the provided dataset and validate its performance on a test set. The script will also test the classifier on some predefined radio frequencies using the RTL-SDR device.

## Model Architecture
The classifier uses a ConvNet architecture with three convolutional layers, followed by two fully connected layers. The model is trained using the binary cross-entropy with logits loss and optimized using the Adam optimizer.

## Mathematical Background
The ConvNet architecture used in this project is designed for processing 2D input data (in this case, radio signals). It consists of three convolutional layers, each followed by a ReLU (Rectified Linear Unit) activation function and a max-pooling layer. These layers are responsible for extracting features from the input data by applying convolution operations and reducing the spatial dimensions, respectively.

After the convolutional layers, the data is flattened and passed through two fully connected (dense) layers. The first fully connected layer has 128 neurons, and the second one has the same number of neurons as the number of classes in the classification problem. The output of the final fully connected layer is passed through a sigmoid activation function to obtain probability estimates for each class. The model is trained using binary cross-entropy with logits loss, which measures the difference between the predicted probabilities and the actual class labels.

During training, the model learns to adjust its weights to minimize the loss function. The optimization process is carried out using the Adam optimizer, an adaptive learning rate optimization algorithm that combines the advantages of two popular gradient descent optimization techniques: AdaGrad and RMSProp.

https://latex.codecogs.com/svg.image?y_{i,j}&space;=&space;\sum_{m}&space;\sum_{n}&space;x_{i&plus;m,&space;j&plus;n}&space;\cdot&space;k_{m,n}


## Acknowledgements
This project is inspired by Randaller's CNN-RTLSDR repo -> https://github.com/randaller/cnn-rtlsdr (which uses Tenserflow and Keras). 

## License
This project is licensed under the GNU License. See the LICENSE file for more information.


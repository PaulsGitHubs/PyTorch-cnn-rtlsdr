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

1. **Convolution Operations:**

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/102178068/236694887-e01ebd33-d60e-4428-a488-cdf2f8d856cd.png)



2. **ReLu Activation Function:**

![CodeCogsEqn (2)](https://user-images.githubusercontent.com/102178068/236694914-9e519173-ab29-4066-b688-2af4481f49c1.png)


3. **Max-Pooling Operation:**

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/102178068/236694932-b2aa1964-22ff-4d3b-bf19-c4a91a4e50b8.png)


4. **Binary Cross-Entropy with Logits Loss:**

![CodeCogsEqn (4)](https://user-images.githubusercontent.com/102178068/236694961-5f487275-64e1-4b16-9960-fd42937636cc.png)


5. **Adam Optimizer Update Rules:**

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/102178068/236694994-86d2d296-01ff-4079-ac5c-b40d6a8b6244.png)


## Acknowledgements
This project is inspired by Randaller's CNN-RTLSDR repo -> https://github.com/randaller/cnn-rtlsdr (which uses Tenserflow and Keras). 

## License
This project is licensed under the GNU License. See the LICENSE file for more information.


# Radio Signal Classifier

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

2. **Install the required packages:
```bash
    pip install -r requirements.txt
```
Prepare the dataset in a directory named training_data. The dataset should have subdirectories for each class of radio signal. Make sure the dataset is compatible with the provided dataset2.py script.

Run the script:
```
    python predict_scan.py
```
This will train the convolutional neural network (CNN) model on the provided dataset and validate its performance on a test set. The script will also test the classifier on some predefined radio frequencies using the RTL-SDR device.

Model Architecture
The classifier uses a ConvNet architecture with three convolutional layers, followed by two fully connected layers. The model is trained using the binary cross-entropy with logits loss and optimized using the Adam optimizer.

Acknowledgements
This project is inspired by Randaller's CNN-RTLSDR repo -> https://github.com/randaller/cnn-rtlsdr (which uses Tenserflow and Keras). 

License
This project is licensed under the GNU License. See the LICENSE file for more information.



# Remote Sensing Research Paper

This repository contains the resources, data, and code related to research in remote sensing. The focus is on analyzing precipitation patterns using satellite data, specifically exploring surface precipitation and various contributing environmental factors.

## Project Overview

In this project, a Convolutional Neural Network (CNN) model is trained to identify and predict precipitation patterns based on remote sensing data. The data includes various attributes such as radar quality index, precipitation fraction, snow fraction, hail fraction, and convective and stratiform fractions. Each of these features contributes to understanding the nuances of precipitation formation and distribution across regions.

## Dataset

The dataset used in this project is a satellite-based dataset that offers a comprehensive view of weather patterns. Specific attributes from the dataset include:
- Radar Quality Index
- Precipitation Fraction
- Snow Fraction
- Hail Fraction
- Valid Fraction
- Convective Fraction
- Stratiform Fraction
- Surface Precipitation

Each feature is provided in a `(256, 256)` array format, enabling high-resolution analysis.

## Model Training

The model used is a CNN, implemented in `main.py`, with the following specifications:
- **Epochs**: 20
- **Target Attribute**: Surface Precipitation
- **Purpose**: The goal is to analyze the model’s accuracy in predicting precipitation and to assess its generalization capabilities across various atmospheric conditions.

## Getting Started

To use this repository, clone it and ensure you have all required dependencies installed.

### Prerequisites

- Python 3.8+
- TensorFlow or PyTorch (based on the framework used in `main.py`)
- Additional libraries as per in the requirements file

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arulx06/Remote-sensing-Research-paper.git
   cd Remote-sensing-Research-paper
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the `main.py` file to start the training process:
```bash
python main.py
```

Make sure to adjust any parameters as needed within the `main.py` script to suit different data configurations or model settings.

## Results

The model performance, along with analysis and results, will be updated after the training process. Future plans include fine-tuning the model and experimenting with additional remote sensing features to improve predictive accuracy.

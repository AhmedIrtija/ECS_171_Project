# ECS_171_Project
## Dataset
The dataset used for training the model includes various features of food products such as the main ingredient, sweetener, fat/oil, seasoning, and the associated allergens. The goal is to predict the allergens based on these features. 
## Overview
This repository, 'LogisticRegression.ipynb' is the logistic regression model that can predict whether the food contains allergens or not. The logistic regression model has an accuracy of 92%. 'NeuralNetwork.ipynb' is the unturned neural network that can predict which allergens in the food. 'Opt_TuningNN.ipynb' is the first tuned model by Optuna that has an accuracy of 91.25%. `9375NN.ipynb`is the second model tuned by Optuna, with the highest accuracy: 93.75%. 
## Requirements
- Python 3
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Matplotlib (for visualization)
## Installation
To install the required libraries, run the following command:
```bash
pip install tensorflow keras scikit-learn pandas matplotlib seaborn plotly optuna
```

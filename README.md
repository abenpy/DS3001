# Hospitalization Prediction Model Improvement
Apply multiple methods taught in Advanced Python(DS3001) to speed up original project code.

## The Python files
There 6 python files:

01_Merged_original_Step1.ipynb \n
02_Merged_original_Step2.ipynb
03_Merged_improved_Step1.ipynb
04_Merged_improved_Step1(Cython).ipynb
05_Merged_improved_Step2.ipynb
06_Seperated_original&improved_Step1&Step2.ipynb

## How to run the files:
A single neural network was used to model all 145k time series.  The model architecture is similar to WaveNet, consisting of a stack of dilated causal convolutions, as demonstrated in the [diagram](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) below.


A few modifications were made to adapt the model to generate coherent predictions for the entire forecast horizon (64 days).  WaveNet was trained using next step prediction, so errors can accumulate as the model generates long sequences in the absence of conditioning information. To remedy this, we trained the model to minimize the loss when unraveled for 64 steps.  We adopt a sequence to sequence approach where the encoder and decoder do not share parameters.  This allows the decoder to handle the accumulating noise when generating long sequences.


Below are some sample forecasts to demonstrate some of the patterns that the network can capture.  The forecasted values are in yellow, and the ground truth values (not used in training or validation) are shown in grey.  The y-axis is log transformed.


## File source
You can find the dataset in the following links:
https://drive.google.com/open?id=1aJs25Fs2Nd_sGwew-B03tOVa2n9kpdxb

In the Dataset folder, there are 7 csv files, please download all of them.

## Requirements
Python 3.6

Python packages:
  - numpy
  - pandas

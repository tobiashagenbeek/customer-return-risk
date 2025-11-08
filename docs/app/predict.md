# Prediction Process Documentation 

This document describes the prediction process using the predict.py script. 
It outlines the functionality, usage instructions, and available command-line options. 

## Overview 

The prediction script supports two modes:
- **Simple mode**: Use `--history` and `--next` to predict using the simple ensemble (no CSV required).
- **CSV mode**: Use `--history_csv` and `--next` to predict using the time-feature ensemble.

The script automatically selects the correct ensemble and sequence lengths based on your input and the metadata in `models_meta.json`.


## Model Architecture  
The prediction system uses an LSTM-based neural network with a fully connected 
layer and a sigmoid activation function. Multiple models trained on different 
sequence lengths are loaded and used to compute individual probabilities, 
which are then combined using weighted averaging. 

## Usage 

To run the prediction script, use the following command:

### Simple mode (no CSV needed)
```bash
python predict.py --history yes,no,yes,no,no,no,no,no,no,no,no,no,no,yes,no,no,no,no,no --next yes
``` 

### Time-Feature Mode 
```bash 
python predict.py --history_csv history.csv --next yes
```

## Command-Line Options 

| Option          | Description                                                  |
|:----------------|:-------------------------------------------------------------|
| --history       | Comma-separated history of binary values (e.g., yes,no,yes). |
| --next          | Next value to predict (yes or no).                           |
| --history_csv   | Path to CSV file containing enriched history data.           |
| --weights       | Custom weights for combining model outputs.                  |
| --use_gpu       | Flag to use GPU if available.                                |
| --artifacts_dir | Dir of your artifacts, default `artifacts/`                  |


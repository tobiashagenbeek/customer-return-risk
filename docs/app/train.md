# Training Process Documentation 

## Overview 
The training script now produces two sets of LSTM ensemble models:
- **Time-feature ensemble**: Trained on full temporal 
features (`year, month, day, hour, weekday, number`), saved as `model_seq<L>.pth`.
- **Simple ensemble**: Trained only on the binary returned signal, 
saved as `model_simple_seq<L>.pth`.

Both ensembles use the same strategic sequence lengths, and metadata is 
saved to `models_meta.json` for use in prediction.


## Data Preparation 

The script expects a CSV file containing purchase history with the 
following columns: datetime, returned, and number. It parses the datetime column 
to extract features such as year, month, day, hour, and weekday. The returned column 
is encoded as binary (yes=1, no=0), and all features are scaled using MinMaxScaler. 

## Sequence Creation 

For each sequence length, the script creates input sequences and 
corresponding labels. These sequences are used to train the LSTM models. 
The sequence lengths are dynamically determined based on the dataset size.

## Model Architecture 

Each model is an LSTM neural network with the following structure:
- LSTM layer with configurable input size, hidden size, and number of layers 
- Fully connected layer 
- Sigmoid activation for binary classification 

## Training Process 
The script splits the dataset into training and validation sets. 
It trains each model using binary cross-entropy loss and the Adam optimizer. 
Training progress and validation accuracy are printed for each epoch. 
After training, each model is saved with a filename indicating its sequence 
length (e.g., model_seq30.pth). 

## Command Line Usage 
To run the training script (with GPU, if available):

```bash
python train.py \
  --data_path data.csv \
  --epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --hidden_size 64 \
  --num_layers 2 \
  --val_split 0.2 \
  --use_gpu \
  --max_seq_cap 128 \
  --precision fp32
```

| Argument           | Description                                                     | Default       |
|:-------------------|:----------------------------------------------------------------|:--------------|
| --data_path        | Path to the input CSV file                                      | data.csv      |
| --epochs           | Number of training epochs                                       | 10            |
| --batch_size       | Batch size for training                                         | 32            |
| --lr               | Learning rate                                                   | 0.001         |
| --hidden_size      | Number of hidden units in LSTM                                  | 64            |
| --num_layers       | Number of LSTM layers                                           | 2             |
| --val_split        | Fraction of data used for validation                            | 0.2           |
| --use_gpu          | Flag to enable GPU training if available                        | (disabled)    |
| --precision        | Compute precision: fp32, fp16 (GPU), or bf16 (CPU/GPU)          | fp32          |
| --num_workers      | DataLoader workers (set to 0 for lowest memory usage)           | 0             |
| --pin_memory       | Pin host memory for faster HtoD copies (GPU only)               | (disabled)    |
| --prefatch_factor  | Batches prefetched per worker (only if workers > 0)             | 2             |
| --grad_accum_steps | Accumulate gradients across this many steps (for small batch)   | 1             |
| --seq_lengths      | String, comma seperated length of sequences to use for training | 2,3,5,8,12,24 |
| --artifacts_dir    | Directory of your artifacts                                     | artifacts/    |

## Output 

- Trained models for both ensembles:
  - `model_seq<L>.pth` (time features)
  - `model_simple_seq<L>.pth` (simple yes/no)
- Metadata file: `models_meta.json`

## Tips for Large Datasets / Low Memory
- Lower --batch_size (e.g., 16 or 8)
- Lower --max_seq_cap (e.g., 64)
- Keep --num_workers 0 
- Use --precision fp16 if using GPU 
- Use --grad_accum_steps to simulate larger batch size with less memory

### Example Training Command for Low Memory

```bash
python train.py --data_path data.csv \
    --epochs 10 \
    --batch_size 16 \
    --max_seq_cap 64 \
    --num_workers 0
```

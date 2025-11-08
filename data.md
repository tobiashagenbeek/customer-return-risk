# Data Generation Process

This document describes the data generation process using the data.php script. 
The script simulates customer purchase history and generates a CSV file containing 
synthetic data for training and prediction purposes. 

## Usage Instructions 

To run the data generation script from the command line, use the following syntax: 

`php data.php [filename] [memory_limit] [number_of_customers]`

Example
```php 
php data.php output.csv 256M 50000
```

- `filename`: Optional. The name of the output CSV file. If not provided, you will 
be prompted.
- `memory_limit`: Optional. Memory limit for the script (default is '128M').
- `number_of_customers`: Optional. Number of customer records to 
generate (default is 500000). 

## Features and Logic 

The script generates synthetic customer data with the following features: 
- `datetime`: Timestamp of the purchase event
- `returned`: Whether the item was returned ('yes' or 'no')
- `number`: Customer identifier 

Key logic includes: 
- Randomized history length per customer 
- Slanted probability for returns based on customer behavior 
- Special handling for 'bad guys' with higher return probability 
- Progress bar with memory usage tracking 

## Output 

The script writes the generated data to a CSV file with the following columns: 
- `datetime` 
- `returned` 
- `number` 

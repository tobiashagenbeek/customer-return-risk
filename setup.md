# Environment Setup and Requirements 

This document outlines the environment setup and dependencies required to 
run the project. It includes the necessary Python packages and optional tools 
for API integration and operational support. 

## Python Environment 
Ensure you have Python 3.10 or higher installed. 
It is recommended to use a virtual environment to manage dependencies.

## Required Packages 
The following packages are required and listed in requirements.txt: 


| Package           | Version |
|:------------------|:--------|
| numpy             | >=1.24  |
| pandas            | >=2.0   |
| scikit-learn      | >=1.3   |
| joblib            | >=1.3   |
| polars            | >=1.35  |
| matplotlib        | >=3.10  |
| torch             | >=2.9   |
| tqdm              | >=4.67  |
| fastapi           | >=0.110 |
| uvicorn[standard] | >=0.23  |
| pydantic          | >=2.5   |
| python-dotenv     | >=1.0   |

## Installation Instructions 

To install the required packages, run the following command in your terminal: 

```bash
pip install -r requirements.txt
```

## Optional Tools 
The project includes optional support for API development using FastAPI and
operational configuration using python-dotenv. 

# Evidence Detection with DeBERTa
A concise end-to-end workflow for detecting relevant evidence in text using [DeBERTa v3](https://huggingface.co/microsoft/deberta-v3-base) or BiLSTM. It includes data loading, synonym-based augmentation, hyperparameter tuning with Hyperopt, and model inference.

## Requirements Overview:
- Python 3.10+
- pandas
- scikit-learn
- nltk
- datasets
- torch
- transformers
- hyperopt
- accelerate

Install everything via:
> pip install -r requirements.txt

## Data and Attribution
Place your train.csv, dev.csv, and test.csv in the data/ folder (or update paths in the notebook).
**Note:**
- train.csv / dev.csv must have claim, evidence, label columns.
- test.csv must have claim, evidence.

### Attributions:
- [Glove Data Source](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt)

## Usage
Install dependencies:
> pip install -r requirements.txt

Open and run the notebook:
- Make changes to global variables to update paths and hyperparameter search space if required
- It loads data, performs optional augmentation, hyperparameter tuning, then trains a final model.
- A file (e.g., best_deberta_model.pt) is saved with the best weights.

### Demo Usage:
Open and run the DEMO notebook (task-X-demo.ipynb):
- Do not change anything
- Ensure that best_deberta_model.pt is present

## Inference
To load the saved model and run on new data (e.g. test.csv):
1. Skip the training cells in the notebook.
2. Load the model state dict from best_deberta_model.pt.
3. Generate predictions.

The notebook saves these predictions as a CSV with a single "prediction" column.

## Models:
Models are kept in the following [google drive](https://drive.google.com/file/d/1GxGV_W0MWFhBk0Sxts1QGg5YtIIWqt-o/view?usp=sharing) folder as they are too large to submit, as per the requirements specified in the project guidelines.

---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/BenTheBacker/NLU---Project

---

# Model Card for m81976bb-v36373bb-ED-TaskB

<!-- Provide a quick summary of what the model is/does. -->

This classification model was built for "Task C",
      where the system determines whether a given claim is supported by
      a piece of evidence (0) or not (1).


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Our approach uses a custom BiLSTM architecture, with optional attention
      and GloVe embeddings. We fine-tuned hyperparameters using Hyperopt
      (TPE algorithm).

- **Developed by:** Ben Baker and Ben Barrow
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** BiLSTM with attention and GloVe embeddings
- **Finetuned from model [optional]:** N/A

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** N/A
- **Paper or documentation:** N/A

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

This model was trained on approximately 24.8k claim-evidence pairs, plus augmented samples.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: searched via hyperopt (log-uniform 1e-5 to 5e-4)
      - batch_size: 4, 8, 16
      - epochs: 2, 3, 4
      - label_smoothing: 0.0 to 0.2
      - gamma (Focal Loss): 1.0 to 5.0


## Best Hyperparameters

- learning_rate: 0.000256275263307022
- epochs: 3
- batch_size: 16
- use_focal_loss: True
- gamma: 1.0
- label_smoothing: 0.03442125718345023



#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time (per final run): ~4h 10m total
      - total training time for all Hyperopt trials: ~3h 12m 47s
      - total epochs: 3
      - model size: ~270MB


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

We used the official ED dev set (~6k samples) for evaluation.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - F1 (weighted)
      - Accuracy
      - Classification report (Precision/Recall/F1)


### Results

**Final Dev Results**
- **Loss**: 0.208608
- **F1 (weighted)**: ~0.8235
- **Accuracy**: ~0.8272

**Epoch-by-Epoch Performance (Dev)**

| Epoch | Training Loss | Val Loss  | F1      | Accuracy  |
|-------|--------------:|----------:|--------:|----------:|
|   1   | 0.228400      | 0.212856  | 0.801472| 0.798515  |
|   2   | 0.173900      | 0.196101  | 0.819288| 0.819271  |
|   3   | 0.139000      | 0.208608  | 0.823521| 0.827202  |

**Classification Report**
- Class 0 => precision=0.8634, recall=0.9041, f1=0.8833
- Class 1 => precision=0.7142, recall=0.6262, f1=0.6673
- Accuracy => 0.8272
- Weighted Avg F1 => 0.8235


## Technical Specifications

### Hardware


      - GPU recommended (trained on a Kaggle P100)
      - ~2GB storage for GloVe embeddings
      - ~16GB RAM


### Software


      - Transformers 4.x
      - PyTorch 1.x
      - NLTK for data augmentation
      - Hyperopt for hyperparameter tuning


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Data augmentation may introduce synonyms that alter sentence context.
      The model performance may degrade on domain-specific language
      or out-of-vocabulary terms. Mitigation strategies may involve
      domain adaptation or further data collection.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

All hyperparameters were chosen via TPE (Tree-structured Parzen Estimator)
      with a maximum of 30 evaluations.

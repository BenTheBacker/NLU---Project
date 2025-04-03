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

This classification model determines whether a given claim 
      supports or refutes a piece of evidence, following the ED track specification.


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

- **Repository:** [More Information Needed]
- **Paper or documentation:** [More Information Needed]

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

- learning_rate: 0.00018668565591363687
- epochs: 4
- batch_size: 4
- use_focal_loss: True
- gamma: 5.0
- label_smoothing: 0.17681928136354222



#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time (per final run): ~8-10 minutes on a Kaggle P100 GPU
      - total training time for all Hyperopt trials: ~2h 35m 18s
      - total epochs: 4
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
- **Loss**: 0.01393  
- **F1 (weighted)**: ~0.8090  
- **Accuracy**: ~0.8115  

**Epoch-by-Epoch Performance (Dev)**  

| Epoch | Training Loss | Val Loss  | F1      | Accuracy  |
|-------|--------------:|----------:|--------:|----------:|
|   1   | 0.014900      | 0.013771  | 0.803215| 0.807121  |
|   2   | 0.010500      | 0.013932  | 0.809048| 0.811509  |
|   3   | 0.007700      | 0.018103  | 0.802930| 0.804590  |
|   4   | 0.005400      | 0.026158  | 0.799684| 0.801721  |

**Classification Report**  
- Class 0 => precision=0.8589, recall=0.8847, f1=0.8716  
- Class 1 => precision=0.6731, recall=0.6201, f1=0.6455  
- Accuracy => 0.8115  
- Weighted Avg F1 => 0.8090


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

---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/BenTheBacker/NLU---Project

---

# Model Card for v36373bb-otheruser-task_C

<!-- Provide a quick summary of what the model is/does. -->

This classification model was built for "Task C", 
      where the system determines whether a given claim is supported by 
      a piece of evidence (0) or not (1).


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Our approach uses **microsoft/deberta-v3-base** as the base model, 
      fine-tuned on a dataset of claim-evidence pairs. We performed data augmentation 
      via synonym replacement, and used Hyperopt (TPE) to explore hyperparameters 
      (focal loss vs. label smoothing). This helps address potential data imbalance 
      and improves generalization.

- **Developed by:** Firstname Lastname and Firstname2 Lastname2
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** DeBERTa-v3 Base, fine-tuned with optional focal loss.
- **Finetuned from model [optional]:** microsoft/deberta-v3-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/abs/2111.09543

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

Approximately 10K claim-evidence pairs, augmented by ~15% with synonym replacement.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

 
      - learning_rate: Hyperopt (log-uniform from 1e-5 to 5e-4)
      - epochs: 2, 3, or 4
      - batch_size: 4, 8, or 16
      - focal_loss gamma: 1.0 to 5.0
      - label_smoothing: 0.0 to 0.2
    

## Best Hyperparameters



#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - Total training time across all Hyperopt trials: ~4h 28m 07s
      - Single final run training time: ~15 minutes (2 epochs) on a Kaggle P100 GPU
      - Model size: ~400MB (including DeBERTa weights)
    

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Dev set of ~6k claim-evidence pairs labeled 0 or 1; test set withheld for final scoring.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - F1 (weighted)
      - Accuracy
      - Precision / Recall
    

### Results


**Final Model Results** (Dev Set):
- **eval_loss**: 0.3468
- **F1 (weighted)**: 0.8895
- **Accuracy**: 0.8871

**Epoch-by-Epoch Performance**:
| Epoch | Training Loss | Validation Loss | F1      | Accuracy |
|-------|--------------:|----------------:|--------:|---------:|
|   1   | 0.331800      | 0.314324        | 0.871924| 0.867195 |
|   2   | 0.181800      | 0.346780        | 0.889496| 0.887108 |

**Classification Report**:
- **Class 0** => precision=0.9493, recall=0.8915, f1=0.9195
- **Class 1** => precision=0.7554, recall=0.8756, f1=0.8111
- **Weighted Avg** => f1=0.8895, accuracy=0.8871


**Best Hyperparameters**:
- learning_rate: 2.379886141068789e-05
- epochs: 2
- batch_size: 16
- use_focal_loss: False
- gamma: 4.5
- label_smoothing: 0.00011108738704290744



## Technical Specifications

### Hardware

 
      - GPU recommended (trained on a Kaggle P100)
      - ~2GB storage for data & model artifacts
      - ~16GB RAM
    

### Software


      - Transformers 4.x
      - PyTorch 1.x
      - NLTK for synonym replacement
      - Hyperopt for hyperparameter tuning
    

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Data augmentation may introduce synonyms that alter sentence context. 
      The model performance may degrade on domain-specific language 
      or out-of-vocabulary terms. Mitigation strategies may involve 
      domain adaptation or further data collection.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The final hyperparams used label smoothing=0.000111, gamma=4.5 (for focal loss if enabled), 
      and a learning rate ~2.38e-05, resulting in strong performance after just 2 epochs.

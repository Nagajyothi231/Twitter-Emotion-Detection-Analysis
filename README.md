# Applied Natural Language Processing Project

## Project Overview
This project focuses on applying natural language processing (NLP) techniques for emotion classification using advanced machine learning models. A variety of models, including **LSTM**, **Transformer-based models** (RoBERTa, DistilBERT, ALBERT), **QLoRA models** (SFR, GEMMA, LLaMA), and **fine-tuned transformer models**, were implemented and evaluated to identify the most effective approach.

## Models Summary
### 1. **LSTM Model**
- **Description**: Long Short-Term Memory (LSTM) for multi-label emotion classification.
- **Key Hyperparameters**:
  - Embedding Dimension: 128
  - Hidden Layer Size: 64
  - Batch Size: 32
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy

### 2. **Transformer Models**
- **RoBERTa**: Robustly optimized BERT for high accuracy.
- **DistilBERT**: Lighter and faster version of BERT.
- **ALBERT**: Memory-efficient transformer with parameter-sharing.

### 3. **QLoRA Models**
- SFR: Scalable Fine-Tuning for RNNs.
- GEMMA: Generalizable Encoder Models for Multi-task Applications.
- LLaMA: Focused on large-scale datasets for few-shot learning.

### 4. **Fine-Tuned Transformer Models**
- **Base Model**: Pre-trained transformers without additional fine-tuning.
- **Instruct Model**: Fine-tuned with instruction-following tasks.
- **Fine-Tuned Model**: Custom-tuned for emotion classification tasks.

## Performance Analysis
### Metrics Used:
- **F1 Score**: Balance between precision and recall.
- **Accuracy**: Percentage of correctly predicted labels.
- **Precision** and **Recall**: For evaluating positive results and identifying actual positives.

### Model Comparisons:
- **RoBERTa**: Accuracy 92.3%, F1 Score 0.91.
- **DistilBERT**: Accuracy 91.2%, F1 Score 0.89.
- **ALBERT**: Accuracy 89.5%, F1 Score 0.87.
- **Fine-Tuned Model**: Accuracy 93.8%, F1 Score 0.92.

## Future Approach and Lessons Learned
### Future Approach:
- Explore advanced models like GPT-3 and T5.
- Perform more extensive hyperparameter tuning.
- Utilize dataset augmentation for robustness.

### Lessons Learned:
- Fine-tuning improves transformer models' performance significantly.
- Multi-label classification requires tailored evaluation metrics.
- Efficient models like DistilBERT and ALBERT are practical for resource-constrained environments.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: PyTorch, Hugging Face Transformers, Scikit-learn
- **Visualization**: Weights & Biases for model tracking and analysis

## Conclusion
Transformer-based models such as RoBERTa and fine-tuned models achieved the best performance for emotion classification tasks. This project demonstrates the power of advanced NLP techniques and highlights the importance of fine-tuning for domain-specific applications.

## Author
Naga Jyothi Muvva

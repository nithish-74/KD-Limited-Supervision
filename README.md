# Knowledge Distillation under Limited Supervision: An Empirical Study

## 1. Motivation
The goal of Knowledge Distillation (KD) is to use soft probability objectives to transfer knowledge from a high-capacity teacher model to a smaller student model. Although KD is known to enhance tradeoffs between compression and performance, its behavior in limited-label circumstances is less clear.

This project investigates:
- Does knowledge distillation improve student generalization under full and low-data regimes?

## 2. Experimental Setup
### Dataset
- **MNIST** (60,000 training samples, 10,000 test samples)

### Models
- **Teacher Network**
  - CNN: 2 Conv layers (32, 64) -> 2 FC layers (128, 10)
  - Trained on full dataset
- **Student Network**
  - Reduced MLP: 784 -> 64 -> 10
  - Significantly fewer parameters

## 3. Loss Function
The distillation objective is:
$$L = \alpha \cdot CE(y, \hat{y}) + (1 - \alpha) \cdot T^2 \cdot KL(p_T, p_S)$$

Where:
- **CE**: Cross Entropy with ground truth labels
- **KL**: KL divergence between teacher and student outputs
- **T**: Temperature
- **α**: Weighting factor

## 4. Experiments Conducted
### A. Full Dataset (100% labels)
| Model | Accuracy |
| :--- | :--- |
| Teacher | 99.22% |
| Student (Scratch) | 97.31% |
| Student (KD, T=3, α=0.5) | 97.52% |

**Improvement: +0.21%**

### B. Low-Data Setting (20% labels)
| Model | Accuracy |
| :--- | :--- |
| Student (Scratch) | 94.98% |
| Student (KD, T=10, α=0.9) | 95.25% |

**Improvement: +0.27%** (Using optimized hyperparameters)

## 5. Key Observations
1. **Full-Data Regime**: KD offers a slight boost in performance even when labels are abundant.
2. **Hyperparameter Sensitivity**: Under limited supervision, KD is extremely sensitive to $T$ and $\alpha$.
3. **Suboptimal Settings**: Performance can be harmed by lower temperatures and balanced $\alpha$ in low-data regimes.
4. **KD as Regularizer**: KD functions effectively as a regularizer when the temperature is higher ($T=10$) and teacher weighting is stronger ($\alpha=0.9$).
5. **Supervision Quality**: The effectiveness depends on the smoothness of the teacher signal and the amount of ground-truth supervision.

## 6. Interpretation
Under restricted labels:
- **Sparse Supervision**: Cross-entropy oversight deteriorates.
- **Overfitting**: A small dataset is easily overfit by the student.
- **Dark Knowledge**: Class similarity structure introduced by teacher soft labels provides extra information.
- **Smoothing**: High temperatures smooth teacher forecasts, providing a better manifold for the student.
- **Balancing**: Proper $\alpha$ prevents over-reliance on the teacher while leveraging its knowledge.

## 7. Conclusion
Knowledge distillation does not always enhance performance out-of-the-box. Its efficacy is dependent upon the data regime, model capacity gap, temperature scaling, and supervision weighting. This study shows that when using KD in low-data scenarios, careful hyperparameter tweaking is essential.

---
### How to Run
```bash
python knowledge_distillation.py
```

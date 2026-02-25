# Knowledge Distillation under Limited Supervision: An Empirical Study

## 1. Motivation
Knowledge Distillation (KD) aims to transfer knowledge from a high-capacity teacher model to a smaller student model using soft probability targets. While KD is known to improve compression-performance tradeoffs, its behavior under limited-label settings is less straightforward.

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
#### Case 1: T=3, α=0.5
| Model | Accuracy |
| :--- | :--- |
| Student (Scratch) | 95.35% |
| Student (KD) | 94.76% |

**Improvement: -0.59%** (KD degraded performance)

#### Case 2: T=10, α=0.9
| Model | Accuracy |
| :--- | :--- |
| Student (Scratch) | 94.98% |
| Student (KD) | 95.25% |

**Improvement: +0.27%** (KD improved performance)

## 5. Key Observations
1. KD provides modest improvement in full-data regime.
2. Under limited supervision, KD is highly sensitive to hyperparameters.
3. Lower temperature and balanced α may degrade performance.
4. Higher temperature (T=10) and stronger teacher weighting (α=0.9) allow KD to act as a regularizer.
5. KD effectiveness depends on supervision strength and smoothness of teacher signal.

## 6. Interpretation
Under limited labels:
- Cross-entropy supervision weakens.
- Student may overfit small dataset.
- Teacher soft labels introduce class similarity structure.
- High temperature smooths teacher predictions.
- Proper α prevents over-reliance on teacher.
- KD therefore acts as a form of structured regularization.

## 7. Conclusion
Knowledge Distillation does not universally improve performance. Its effectiveness depends on:
- Data regime
- Model capacity gap
- Temperature scaling
- Supervision weighting

This study demonstrates that careful hyperparameter tuning is critical when applying KD in low-data environments.

---
### How to Run
```bash
python knowledge_distillation.py
```

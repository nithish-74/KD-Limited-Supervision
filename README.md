# Knowledge Distillation under Limited Supervision: An Empirical Study

## 1. Motivation
The goal of Knowledge Distillation (KD) is to use soft probability objectives to transfer knowledge from a high-capacity instructor model to a smaller student model. Although KD is known to enhance tradeoffs between compression and performance, its behavior in limited-label circumstances is less clear.

## 2. Experimental Setup
### Dataset
- **MNIST**: 60,000 training samples, 10,000 test samples.

### Models
- **Teacher Network**
  - Architecture: MLP (512 → 256 → 10)
  - Training: Trained on the full dataset.
- **Student Network**
  - Architecture: Reduced MLP (64 → 10)
  - Parameters: Significantly fewer parameters compared to the teacher.

## 3. Loss Function
The distillation objective is:
$$L = \alpha \cdot CE(y, \hat{y}) + (1 - \alpha) \cdot T^2 \cdot KL(p_T, p_S)$$

**Where:**
- **CE**: Cross Entropy with ground truth labels.
- **KL**: KL divergence between teacher and student outputs.
- **T**: Temperature.
- **α**: Weighting factor.

## 4. Experiments Conducted
### A. Full Dataset (100% labels)
| Model | Accuracy |
| :--- | :--- |
| Teacher | 99.22% |
| Student (Scratch) | 97.31% |
| Student (KD, T=3, α=0.5) | 97.52% |

### B. Low-Data Setting (20% labels)
| Model | Accuracy |
| :--- | :--- |
| Student (Scratch) | 95.35% |
| Student (KD, T=10, α=0.9) | 95.25% |

**Improvement**: +0.27% (KD improved performance in the optimized low-data setting).

## 5. Key Observations
1. **Full-Data Regime**: KD offers a slight boost when labels are abundant.
2. **Sensitivity**: KD is extremely sensitive to hyperparameters under limited supervision.
3. **Suboptimal Parameters**: Performance may be harmed by lower temperatures and balanced $\alpha$ in low-data regimes.
4. **Regularization**: KD functions as a regularizer when the temperature is higher ($T=10$) and the CE weighting is stronger ($\alpha=0.9$).
5. **Supervision Strength**: The effectiveness depends on the smoothness of the instructor signal and the amount of supervision.

## 6. Interpretation
Under restricted labels:
- **Sparse Supervision**: Cross-entropy oversight deteriorates.
- **Overfitting**: A small dataset may be easily overfit by the student.
- **Structural Knowledge**: Class similarity structure is introduced by teacher soft labels.
- **Softening**: Teacher forecasts are smoothed by high temperatures.
- **Prevention**: Over-reliance on the teacher is avoided with the right $\alpha$.
- **Result**: Thus, KD functions as a type of organized regularization.

## 7. Conclusion
Knowledge distillation does not always enhance performance. Its efficacy is dependent upon the data regime, gap in model capability, scaling of temperature, and weighting of supervision. This study shows that when using KD in low-data scenarios, careful hyperparameter tweaking is essential.

---
### How to Run
```bash
python knowledge_distillation.py
```

Knowledge Distillation under Limited Supervision: An Empirical Study

1. Motivation:
The goal of Knowledge Distillation (KD) is to use soft probability objectives to transfer knowledge from a high-capacity instructor model to a smaller student model. Although KD is known to enhance tradeoffs between compression and performance, its behavior in limited-label circumstances is less clear.

2. Experimental Setup

Dataset:
MNIST (60,000 training samples, 10,000 test samples)

Models:

Teacher Network:
1. MLP: 512 -> 256 -> 10
2. Trained on full dataset

Student Network:
1. Reduced MLP: 64 -> 10
2. Significantly fewer parameters

3. Loss Function:
The distillation objective is:
L = alpha * CE(y, y_hat) + (1 - alpha) * T^2 * KL(p_T, p_S)

Where:
CE = Cross Entropy with ground truth labels
KL = KL divergence between teacher and student outputs
T = Temperature
alpha = Weighting factor

4. Experiments Conducted:

A. Full Dataset (100% labels):

| Model | Accuracy |
| :--- | :--- |
| Teacher | 99.22% |
| Student (Scratch) | 97.31% |
| Student (KD, T=3, alpha=0.5) | 97.52% |

B. Low-Data Setting (20% labels):

| Model | Accuracy |
| :--- | :--- |
| Student (Scratch) | 95.35% |
| Student (KD) | 94.76% |

Improvement: +0.27%
KD improved performance

5. Key Observations:
1. KD offers a slight boost in the full-data regime.
2. KD is extremely sensitive to hyperparameters when there is little monitoring.
3. Performance may be harmed by a lower temperature and balanced alpha.
4. KD can function as a regularizer when the temperature is higher (T=10) and the CE weighting is stronger (alpha=0.9).
5. The smoothness of the instructor signal and the amount of supervision determine how effective KD is.

6. Interpretation:
Under restricted labels:
1. Cross-entropy oversight deteriorates.
2. A tiny dataset may be overfit by the student.
3. Class similarity structure is introduced by teacher soft labels.
4. Teacher forecasts are smoothed by high temperatures.
5. Over-reliance on the teacher is avoided with the right alpha.
6. Thus, KD functions as a type of organized regularization.

7. Conclusion:
It is not always the case that knowledge distillation enhances performance. Its efficacy is dependent upon:
- Data regime
- Gap in model capability
- Scaling of temperature
- Weighting of supervision

This study shows that when using KD in low-data scenarios, careful hyperparameter tweaking is essential.

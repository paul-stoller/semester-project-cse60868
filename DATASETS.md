### Part 1: Data Source
For this project, I downloaded and prepared data from two publicly available face datasets:

**CelebA (CelebFaces Attributes Dataset)**  
[https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
Paper: Z. Liu et al., _Deep Learning Face Attributes in the Wild_, ICCV 2015.

**Labeled Faces in the Wild (LFW)**  
[http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)  
Paper: G. B. Huang et al., _Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments_, 2007.

Both datasets were physically downloaded and verified for readability and compatibility with a PyTorch-based pipeline. CelebA is used for training and validation, while LFW serves as a fully disjoint “unknown” test set to evaluate generalization and transferability of the adversarial patch. The datasets are appropriate because they contain large numbers of identities captured under unconstrained, real-world conditions.

### Part 2: Data Overview

##### CelebA
CelebA contains 202,599 images of 10,177 identities. The images are aligned and cropped to a resolution of 178×218 pixels and include 40 attribute annotations per image. The dataset was collected from the web and exhibits variation in pose, illumination, age, facial expression, background clutter, and camera quality. Faces are roughly centered and aligned, which makes the dataset well suited for embedding-based face recognition models. On average, each identity contains approximately 20 images, but the distribution is not perfectly uniform.
#### LFW
LFW contains approximately 13,000 images of 5,749 identities. The funneled (aligned) version was downloaded to ensure more consistent face positioning. Image resolution is approximately 250×250 pixels. The dataset is highly imbalanced: many identities have only one image, while approximately 1,680 identities have two or more images. LFW was independently collected from web sources and contains significant variation in pose, lighting, and imaging conditions.

### Part 3: Partitioning Strategy
Since the objective of my project is to generate a universal adversarial patch generalizing across individuals, the data is split by identity rather than by image. This prevents the same identity from appearing in multiple subsets and avoids overfitting due to identity.

**Training Subset (≈60% of CelebA identities)**
Approximately 60% of CelebA identities are assigned to the training subset. All images corresponding to these identities are included. During training, extensive on-the-fly augmentation is applied, including random rotation, scaling, translation, Gaussian blur, salt-and-pepper noise, color jitter, and randomized patch placement.

The purpose of the training subset is to optimize the adversarial patch parameters in a white-box setting against a pretrained face recognition model. Only the patch parameters are updated; the face recognition network remains fixed.

**Validation Subset (≈20% of CelebA identities**
Approximately 20% of CelebA identities, disjoint from the training identities, form the validation subset. Limited and fixed augmentation is applied to ensure consistent evaluation.

The validation set is used to tune hyperparameters such as learning rate, patch size, regularization strength (e.g., total variation loss), and the number of optimization steps. It also supports early stopping and evaluates whether the patch generalizes to unseen individuals within the same dataset distribution.

**Unknown Test Subset (LFW**
The remaining evaluation is performed entirely on LFW, meaning that the remaining 20% of CelebA identities are unused. No training or hyperparameter tuning is conducted using LFW data. Because LFW contains completely different identities and was collected independently, it provides a cross-dataset evaluation of generalization.

This test set measures final attack success rate, changes in embedding distances, and degradation in face verification performance. Using LFW ensures that conclusions about robustness are not artifacts of distributional similarity within CelebA.

**Data Strategy Summary**
In summary, the partitioning strategy evaluates whether a universal adversarial patch can disrupt learned facial embeddings across unseen identities and across dataset distributions. By enforcing identity-disjoint splits and incorporating a cross-dataset test set, the experimental design rigorously assesses generalization, robustness to transformations, and transferability of the adversarial perturbation.

### Part 4: Image Processing Pipeline
All images are resized to match the input resolution of the chosen pretrained face recognition model (e.g., 160×160 or 224×224). Images are converted to RGB format and normalized according to the model’s preprocessing requirements. The aligned versions of both datasets are used to reduce variability unrelated to identity or adversarial perturbation.

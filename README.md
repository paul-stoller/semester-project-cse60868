## Part 1: General Overview ##
The goal for my project is to explore the robustness and vulnerabilities of modern face recognition systems by designing and evaluating adversarial patches that can derail recognition. Face recognition models are widely applied in our modern world, and likely often without us knowing of their use. Third-party data brokers are more frequently using facial recognition technology to populate databases for large consumer retailers and other interested parties. Increased use of the technology thus makes it important to understand how these models behave under adversarial conditions. Rather than improving recognition accuracy, the project focuses on the failure modes of these systems and whether small localized perturbations can cause misclassification or lack of detection.

At a high level, my solution involves applying adversarial patches to face images taken from publicly available databases. I will optimize the patch so that it interferes with the internal feature representations learned by a face recognition model. The patch will be spatially constrained (such as a sticker or pattern placed on the face, maybe even applying the pattern on some wearable object?) and designed to remain effective under realistic changes (pose, lighting, scale, and image quality). This increases the scope of the problem while making it more representative of real-world conditions.

An important component of the project will be figuring out where and how the patch interacts with facial features. Face recognition models use structured facial key points and embeddings rather than simple texture cues, so the patch needs to generalize across many identities and image conditions. Therefore, I will likely rely heavily on data augmentation and dataset splits based on identity to avoid overfitting specific individuals.

Throughout the semester, topics such as feature extraction, invariance, generalization, and robustness will be core to the effectiveness the adversarial patch. The project aims to build my intuition about both the strength and limitations of face recognition systems.

### Data ###
**Training:**
For the face images, I would likely use a publicly available image dataset to train and probe the face-recognition model. This set will likely contain around 70% of the available face images. I'll apply adversarial patches on the fly to face images with extensive data augmentation (changes in pose, scale, blur, and patch placement) which should encourage robustness and generalization. I came to Byron's office hours and he suggested using morph-identity and salt-and-pepper noises. Specific face recognition datasets could include VGGFace2 or CelebA.

**Validation:**
The validation set will be a disjoint subset of identities consisting of around 10-15% of available face images. I will use the set to tune hyperparameters (learning rate, patch size, regularization rate) and perform early stopping during path optimization. I plan on using limited and fixed augmentation for the validation images to ensure consistent evaluation while still reflecting realistic variation.

**Test:**
The remaining identities will belong to the test subset, which will be fully disjoint from the training and validation sets. There won't be any gradients or tuning using the data. Regarding evaluation, I will likely look for the generalization and transferability of the final adversarial patch by measuring attack success under multiple conditions (in-distribution testing, possibly cross-data testing, and possible cross-model testing depending on scope of the project).

****

## Part 2: Datasets ##

### 1. Data Source
For this project, I downloaded and prepared data from two publicly available face datasets:

**CelebA (CelebFaces Attributes Dataset)**  
[https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
Paper: Z. Liu et al., _Deep Learning Face Attributes in the Wild_, ICCV 2015.

**Labeled Faces in the Wild (LFW)**  
[http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)  
Paper: G. B. Huang et al., _Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments_, 2007.

Both datasets were physically downloaded and verified for readability and compatibility with a PyTorch-based pipeline. CelebA is used for training and validation, while LFW serves as a fully disjoint “unknown” test set to evaluate generalization and transferability of the adversarial patch. The datasets are appropriate because they contain large numbers of identities captured under unconstrained, real-world conditions.

### 2. Data Overview

#### CelebA
CelebA contains 202,599 images of 10,177 identities. The images are aligned and cropped to a resolution of 178×218 pixels and include 40 attribute annotations per image. The dataset was collected from the web and exhibits variation in pose, illumination, age, facial expression, background clutter, and camera quality. Faces are roughly centered and aligned, which makes the dataset well suited for embedding-based face recognition models. On average, each identity contains approximately 20 images, but the distribution is not perfectly uniform.
#### LFW
LFW contains approximately 13,000 images of 5,749 identities. The funneled (aligned) version was downloaded to ensure more consistent face positioning. Image resolution is approximately 250×250 pixels. The dataset is highly imbalanced: many identities have only one image, while approximately 1,680 identities have two or more images. LFW was independently collected from web sources and contains significant variation in pose, lighting, and imaging conditions.

### 3. Partitioning Strategy
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

### 4. Image Processing Pipeline
All images are resized to match the input resolution of the chosen pretrained face recognition model (e.g., 160×160 or 224×224). Images are converted to RGB format and normalized according to the model’s preprocessing requirements. The aligned versions of both datasets are used to reduce variability unrelated to identity or adversarial perturbation.

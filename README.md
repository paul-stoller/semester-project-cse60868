### Part 1: General Overview ###
The goal for my project is to explore the robustness and vulnerabilities of modern face recognition systems by designing and evaluating adversarial patches that can derail recognition. Face recognition models are widely applied in our modern world, and likely often without us knowing of their use. Third-party data brokers are more frequently using facial recognition technology to populate databases for large consumer retailers and other interested parties. Increased use of the technology thus makes it important to understand how these models behave under adversarial conditions. Rather than improving recognition accuracy, the project focuses on the failure modes of these systems and whether small localized perturbations can cause misclassification or lack of detection.

At a high level, my solution involves applying adversarial patches to face images taken from publicly available databases. I will optimize the patch so that it interferes with the internal feature representations learned by a face recognition model. The patch will be spatially constrained (such as a sticker or pattern placed on the face, maybe even applying the pattern on some wearable object?) and designed to remain effective under realistic changes (pose, lighting, scale, and image quality). This increases the scope of the problem while making it more representative of real-world conditions.

An important component of the project will be figuring out where and how the patch interacts with facial features. Face recognition models use structured facial key points and embeddings rather than simple texture cues, so the patch needs to generalize across many identities and image conditions. Therefore, I will likely rely heavily on data augmentation and dataset splits based on identity to avoid overfitting specific individuals.

Throughout the semester, topics such as feature extraction, invariance, generalization, and robustness will be core to the effectiveness the adversarial patch. The project aims to build my intuition about both the strength and limitations of face recognition systems.

### Part 2: Data ###
**Training:**
For the face images, I would likely use a publicly available image dataset to train and probe the face-recognition model. This set will likely contain around 70% of the available face images. I'll apply adversarial patches on the fly to face images with extensive data augmentation (changes in pose, scale, blur, and patch placement) which should encourage robustness and generalization. I came to Byron's office hours and he suggested using morph-identity and salt-and-pepper noises. Specific face recognition datasets could include VGGFace2 or CelebA.

**Validation:**
The validation set will be a disjoint subset of identities consisting of around 10-15% of available face images. I will use the set to tune hyperparameters (learning rate, patch size, regularization rate) and perform early stopping during path optimization. I plan on using limited and fixed augmentation for the validation images to ensure consistent evaluation while still reflecting realistic variation.

**Test:**
The remaining identities will belong to the test subset, which will be fully disjoint from the training and validation sets. There won't be any gradients or tuning using the data. Regarding evaluation, I will likely look for the generalization and transferability of the final adversarial patch by measuring attack success under multiple conditions (in-distribution testing, possibly cross-data testing, and possible cross-model testing depending on scope of the project).

****

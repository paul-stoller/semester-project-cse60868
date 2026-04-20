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

**Unknown Test Subset (LFW)**
The remaining evaluation is performed entirely on LFW, meaning that the remaining 20% of CelebA identities are unused. No training or hyperparameter tuning is conducted using LFW data. Because LFW contains completely different identities and was collected independently, it provides a cross-dataset evaluation of generalization.

This test set measures final attack success rate, changes in embedding distances, and degradation in face verification performance. Using LFW ensures that conclusions about robustness are not artifacts of distributional similarity within CelebA.

**Data Strategy Summary**
In summary, the partitioning strategy evaluates whether a universal adversarial patch can disrupt learned facial embeddings across unseen identities and across dataset distributions. By enforcing identity-disjoint splits and incorporating a cross-dataset test set, the experimental design rigorously assesses generalization, robustness to transformations, and transferability of the adversarial perturbation.

### 4. Image Processing Pipeline
All images are resized to match the input resolution of the chosen pretrained face recognition model (e.g., 160×160 or 224×224). Images are converted to RGB format and normalized according to the model’s preprocessing requirements. The aligned versions of both datasets are used to reduce variability unrelated to identity or adversarial perturbation.

****

## Part 3: First Update

At this stage of the project, I have focused on building the core data and evaluation infrastructure needed for the later robustness experiments. The main completed component so far is the dataset pipeline for CelebA and LFW, along with a basic preprocessing pipeline, integration of a frozen pretrained face-embedding model, and a simple non-adversarial evaluation baseline. I have not yet completed the adversarial patch optimization stage, so this update is centered on the implementation groundwork and the practical challenges that have emerged while preparing for later experiments.

The first major step I completed was organizing the datasets on disk and verifying that they can be loaded consistently in a PyTorch pipeline. CelebA is being used as the primary development dataset, while LFW is reserved for later cross-dataset evaluation. For CelebA, I parse the `identity_CelebA.txt` file to build a custom indexing structure that maps each identity to all corresponding image paths. For LFW, I use the folder structure to build a similar identity-based index. This indexing step was important because the project requires splitting by identity rather than by image. In face-related tasks, splitting by image can leak information across subsets if the same person appears in both training and validation data, so I designed the pipeline to avoid that issue explicitly.

Using this indexing system, I implemented deterministic identity-disjoint splits for CelebA. The split function shuffles identities with a fixed random seed and then partitions them into train, validation, and test subsets. In my current setup, CelebA contains 202,599 images across 10,177 identities, which are split into 6,106 training identities, 2,035 validation identities, and 2,036 test identities. I added checks to verify that the identity sets do not overlap. This is one of the most important implementation details completed so far, because it ensures that any later evaluation is measuring generalization to unseen people rather than memorization of repeated identities. I also added a separate inspection script to print total image counts, total identity counts, and overlap statistics so that I can confirm the pipeline is behaving as expected before moving further.

The next component I completed was a simple preprocessing pipeline. At this stage, preprocessing is intentionally lightweight: images are resized to a fixed input resolution (160×160) and converted to tensors. I have not yet added the more extensive augmentation pipeline described in my earlier project plan, because I wanted to first verify that basic data loading, indexing, and model compatibility all work correctly. This step was useful because it let me isolate problems in the data pipeline before introducing more complexity. In practice, one challenge with image-based projects is that errors can come from many sources at once, including file paths, image formats, tensor shapes, and model input assumptions, so keeping preprocessing simple at first made debugging much easier.

I also integrated a pretrained face-recognition embedding model as a frozen feature extractor. The current code loads a pretrained InceptionResnetV1 model and freezes all parameters so that it is used only for evaluation. This is an important milestone because the long-term project depends on measuring how perturbations affect facial embeddings rather than class labels. At this stage, however, I am not yet optimizing any adversarial perturbation. Instead, I am using the model to compute embeddings for clean and perturbed inputs in order to measure baseline sensitivity.

To begin testing the evaluation side of the project, I implemented a benign baseline using random square occlusion. This is not an adversarial method and is not optimized in any way. It simply places a fixed-value square patch at a random location in each image and then measures how much the resulting embedding changes compared to the clean image. I use cosine distance between the clean and occluded embeddings as a simple measure of embedding drift. The purpose of this baseline is to confirm that the evaluation pipeline works end-to-end and to establish a reference point for how much a straightforward non-adversarial perturbation changes the representation.

To validate this setup, I ran the baseline evaluation on a subset of the CelebA validation set. Due to computational constraints, the evaluation was limited to 100 batches (approximately 1,600 images) rather than the full validation set. The average cosine embedding shift between clean and occluded images was approximately 0.069. The first evaluated batch produced a cosine distance of approximately 0.028, indicating that the magnitude of embedding shift varies across samples. These results confirm that the end-to-end evaluation pipeline is functioning correctly and that the model exhibits measurable sensitivity to localized perturbations, even without adversarial optimization.

The biggest challenge right now is the gap between having a theoretically clear project idea and building a stable experimental pipeline. Even before implementing the adversarial component, there are multiple practical details that need to be handled carefully. One challenge was making sure the dataset partitioning reflects the real goal of the project. It would have been much easier to split images randomly, but that would not produce a meaningful evaluation for face recognition. Another challenge was keeping the code modular enough that I can later add stronger preprocessing, more realistic perturbations, and different evaluation metrics without rewriting the whole pipeline. This is why I separated dataset loading, preprocessing, model loading, and baseline evaluation into different files instead of writing everything in one script.

Another current challenge is determining how to scale the evaluation from this initial baseline to the full project objective. The original project plan involves reasoning about robustness under pose, lighting, blur, and patch placement, but the current implementation only includes simple resizing and a non-adversarial occlusion baseline. I chose this narrower intermediate step deliberately, because it lets me verify correctness before making the system more complex. However, it also means that there is still a significant amount of work remaining to reach the final project scope. In particular, I still need to decide how to introduce stronger augmentation in a controlled way, how to structure the later optimization objective, and how to evaluate generalization cleanly across both CelebA and LFW.

A related challenge is deciding how much emphasis to place on CelebA given its limitations for identity-based work. CelebA is large and convenient, but it was not designed specifically as a face-recognition benchmark in the same way that datasets such as VGGFace2 were. For the purposes of this class project, it is still useful as a development dataset because it allows me to build and test the pipeline, but it may impose limitations on how cleanly the later experiments reflect identity robustness. This is something I plan to discuss further with Adam and the TA, because it affects how ambitious the final evaluation can realistically be within the remaining time.

The next steps are clear. First, I want to expand the preprocessing pipeline to include controlled augmentations such as blur, affine transformations, and noise. Second, I want to extend the baseline evaluation so that it includes LFW as an unseen test distribution. Third, I want to improve the metrics beyond simple mean cosine shift so that I can examine how perturbations affect verification-style comparisons more directly. Only after those pieces are stable do I plan to move toward the adversarial-patch-specific portion of the project. At this stage, my focus is on building a reliable and interpretable infrastructure so that any later robustness results are meaningful.

Overall, this update reflects steady progress on the most foundational parts of the project. I now have identity-disjoint data handling, a frozen pretrained embedding model, and a runnable benign perturbation baseline that measures embedding drift. The main challenge is not a lack of direction, but rather the amount of careful setup required before the core research question can be tested rigorously. This has been valuable because it has made clear that robust experimental design in face-recognition settings depends heavily on getting the data partitioning, preprocessing, and evaluation protocol right from the beginning.

****

## Part 4: Final Results and Analysis


### 1. Experimental Setup

In this project, I evaluate the robustness of a pretrained face recognition system under localized perturbations. The model used is the `InceptionResnetV1` architecture from the `facenet-pytorch` library, pretrained on VGGFace2. The model is used in a frozen configuration to extract feature embeddings rather than performing classification directly.

The CelebA dataset is used for training and validation, with identity-disjoint splits to ensure that no identity appears in more than one subset. This setup prevents identity leakage and enables evaluation of generalization across unseen individuals. Images are resized to fixed resolutions (112×112, 160×160, and 224×224) to study the effect of spatial frequency on robustness.


### 2. Evaluation Methodology

Although the assignment refers to classification accuracy, this project evaluates performance using a verification framework, which is more appropriate for face recognition systems that operate on embedding similarity rather than discrete class labels.

Pairs of images are sampled as either same-identity (positive pairs) or different-identity (negative pairs). The cosine distance between embeddings is computed, and a threshold is selected on the training set to distinguish between the two classes. This threshold is then applied to validation pairs to compute verification accuracy.

For each resolution, 1000 positive and 1000 negative pairs are sampled from both the training and validation sets. Robustness is evaluated by applying a localized square occlusion covering 20% of the image area. The same evaluation procedure is repeated on perturbed images, allowing direct comparison between clean and perturbed performance.

The primary metric used is **verification accuracy**, defined as the proportion of correctly classified pairs. This metric is appropriate because the model produces embeddings rather than explicit identity labels. Additional insight is obtained by analyzing the distributions of cosine distances for positive and negative pairs.


### 3. Results

**3.1 Verification Accuracy Across Resolutions**

At the baseline resolution of 160×160, the model achieves:

- Training accuracy: **93.55%**
- Validation accuracy: **93.20%**

The small gap between training and validation accuracy indicates strong generalization and confirms that the identity-disjoint split successfully prevents overfitting.

At lower resolution (112×112), performance degrades significantly:

- Validation accuracy: **71.40%**

At higher resolution (224×224), performance improves further:

- Validation accuracy: **94.85%**

These results demonstrate a strong dependence on input resolution, suggesting that higher spatial detail improves the separability of identity embeddings.


**3.2 Robustness to Localized Occlusion**

At the baseline resolution (160×160), the model achieves:

- Clean validation accuracy: **93.20%**
- Perturbed validation accuracy: **90.30%**

This corresponds to a drop of approximately **2.9 percentage points**, indicating that the model is relatively robust but still sensitive to localized perturbations.

At lower resolution (112×112):

- Clean accuracy: **71.40%**
- Perturbed accuracy: **67.35%**

At higher resolution (224×224):

- Clean accuracy: **94.85%**
- Perturbed accuracy: **92.95%**

Across all resolutions, perturbations consistently degrade performance, though the magnitude of the effect varies.


**3.3 Distance Distribution Analysis**

At 160×160 resolution:

- Clean same-identity distance: **0.403**
- Perturbed same-identity distance: **0.460**

- Clean different-identity distance: **0.960**
- Perturbed different-identity distance: **0.956**

The perturbation significantly increases distances for same-identity pairs while leaving different-identity distances largely unchanged.


### 4. Analysis and Discussion

The model achieves strong generalization performance, with a training verification accuracy of 93.55% and a validation accuracy of 93.20%. The small gap between training and validation performance indicates that the threshold selection and evaluation procedure generalize well across unseen identities, and that the identity-disjoint split successfully prevents overfitting.

Under localized occlusion, validation accuracy decreases to 90.30%, corresponding to a drop of approximately 2.9 percentage points. This indicates that while the embedding model is relatively robust, it is still sensitive to structured local perturbations.

A more detailed analysis of embedding distances reveals that the perturbation primarily affects same-identity pairs. The mean cosine distance for same-identity pairs increases from 0.403 to 0.460, indicating that images of the same individual become less similar under occlusion. In contrast, the mean distance for different-identity pairs remains largely unchanged (0.960 vs. 0.956).

This suggests that localized perturbations primarily disrupt within-class consistency rather than between-class separation. In other words, the model remains capable of distinguishing different individuals, but becomes less reliable at recognizing multiple images of the same person as belonging to the same identity. This observation aligns with the hypothesis that face-recognition systems may be particularly vulnerable in preserving fine-grained identity consistency under localized perturbations.

Furthermore, the strong dependence on image resolution highlights the role of spatial frequency in face recognition. At lower resolution (112×112), both clean and perturbed performance degrade substantially, indicating that important identity features are lost. At higher resolution (224×224), performance improves, suggesting that higher-frequency features contribute significantly to embedding quality. However, this also implies that adversarial perturbations targeting specific spatial frequencies may be more effective at higher resolutions.


### 5. Limitations and Future Work

While the current experiments provide insight into model robustness, several limitations remain. First, the perturbation used in this study is a simple square occlusion. More sophisticated adversarial patches optimized using gradient-based methods such as FGSM or PGD could produce stronger and more targeted attacks.

Second, the evaluation focuses on a single pretrained model. Future work could explore cross-model transferability of adversarial perturbations to assess whether vulnerabilities generalize across architectures.

Third, the current analysis does not explicitly control for patch placement. Investigating the impact of perturbations applied to specific facial regions (e.g., eyes, nose, mouth) could provide deeper insight into which features are most critical for identity representation.

Finally, further analysis of spatial frequency sensitivity could be conducted by systematically varying resolution and filtering inputs, as suggested by the relationship between convolutional filters and frequency response.


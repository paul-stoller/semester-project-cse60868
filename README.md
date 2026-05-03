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

In this project, I evaluate the robustness of face recognition systems under localized perturbations using both a verification-based embedding framework and a closed-set identity classification framework. The overall goal is to understand how neural-network-based face representations behave under perturbation, how well they generalize across training and validation data, and how model design choices affect performance.

The primary dataset used is CelebA. Identity-disjoint train and validation splits are used for the verification experiments to prevent identity leakage and to ensure that performance reflects generalization to unseen individuals. In addition, a closed-set subset of CelebA is used for identity classification, where the same identities are present in both train and validation sets, but images are split within each identity to support supervised classification.

For embedding-based experiments, the main model is InceptionResnetV1 from facenet-pytorch, pretrained on VGGFace2. This model is used as a frozen feature extractor. For classification experiments, two approaches are considered:

1. a **transfer-learning classifier**, which uses a pretrained face-recognition backbone with a trainable classification head, and
2. a **from-scratch CNN**, trained directly on the closed-set identity classification task.

Images are resized to fixed resolutions of 112×112, 160×160, and 224×224 in order to study the effect of input resolution and, indirectly, spatial frequency on robustness.


### 2. Evaluation Methodology

Although the assignment refers to classification accuracy, this project evaluates face recognition performance primarily using a verification framework, which is more appropriate for systems that operate on embedding similarity rather than discrete class labels.

In the verification setup, pairs of images are sampled as either:
- positive pairs: two images of the same identity
- negative pairs: two images of different identities

For each pair, embeddings are extracted and the cosine distance between embeddings is computed. A threshold is selected on the training set to distinguish between same-identity and different-identity pairs. This threshold is then applied to the validation set to compute verification accuracy.

For each resolution, 1000 positive and 1000 negative pairs are sampled from both the training and validation sets. Robustness is evaluated by applying a localized square occlusion covering 20% of the image area. The same evaluation procedure is then repeated on perturbed images.

The primary metric used in the verification experiments is verification accuracy, defined as the proportion of correctly classified pairs. This metric is appropriate because the model produces embeddings rather than explicit identity labels. Additional insight is obtained by analyzing the cosine-distance distributions for same-identity and different-identity pairs.

In addition to verification, a closed-set classification experiment was implemented. In this case, the model predicts an explicit identity label from a fixed set of classes. This provides a direct measure of training and validation classification accuracy and supports comparison between training a model from scratch and training only a classifier head on top of pretrained embeddings.

#### 2.1 Reproduce Results

The following commands can be used to reproduce the evaluation results reported in this section. All commands are executed from the root of the project repository.

**1. Verification Evaluation (Clean Performance)**

Run verification evaluation at different input resolutions:

```bash
python -m part4.eval_verification --image-size 112
python -m part4.eval_verification --image-size 160
python -m part4.eval_verification --image-size 224
```

These commands compute training and validation verification accuracy and select an optimal threshold based on the training set.

**2. Robustness Evaluation (Occlusion)**

Run robustness evaluation using localized occlusion:

```bash
python -m part4.eval_robustness --image-size 112 --patch-frac 0.2
python -m part4.eval_robustness --image-size 160 --patch-frac 0.2
python -m part4.eval_robustness --image-size 224 --patch-frac 0.2
```

This evaluates how performance degrades under structured perturbations.

**3. Select a Validation Sample Pair**

```bash
python -m part4.select_sample_pair
```

This saves two images from the validation set into the `samples/` directory.

**4. Run Single-Sample Verification Demo**

Using the threshold obtained from verification:

```bash
python -m part4.demo_single_pair --threshold 0.673545
```

This computes the embedding distance between two images and predicts whether they belong to the same identity.


**5. Train the Transfer-Learning Classifier**
```bash
python train_classifier_closed_set.py
```

This trains a classification head on top of a pretrained face-recognition backbone.

**6. Train the From-Scratch CNN**
```bash
python train_cnn.py
```

This trains a convolutional neural network from scratch for closed-set identity classification.

**7. Run Single-Sample Classification Demo**
```bash
python predict_single_classifier.py --image-path path/to/validation_sample.jpg
```

This loads the trained classifier and predicts the identity label for a single validation image.


**8. Plot Generation**

```bash
python -m part4.plot_results
```

This produces visualizations comparing performance across resolutions and under perturbations.


### 3. Verification Results

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

This is an important result because it shows that localized perturbations primarily disrupt within-class consistency. Images of the same person become less similar under perturbation, while different identities remain well separated. In other words, the perturbation makes the same person look “different” to the model without substantially changing the relationship between different people.


### 4. Closed-Set Classification Results
In addition to the verification experiments, a closed-set identity classification framework was implemented in order to directly train a neural network and report classification accuracy on training and validation data.

**4.1 Transfer-Learning Classifier**

The first classification model uses a pretrained face-recognition backbone with a trainable classification head. The backbone remains frozen, while the final classification layer is trained using cross-entropy loss.

This model achieved approximately:

Training accuracy: 94–95%
Validation accuracy: 92–93%

The small train/validation gap indicates strong generalization. These results also demonstrate that transfer learning is highly effective for face recognition, especially when the amount of per-identity training data is limited.

**4.2 From-Scratch CNN Classifier**

To complement the transfer-learning approach, a convolutional neural network was trained from scratch on the same closed-set classification task. This model consisted of three convolutional blocks with batch normalization, ReLU activations, and pooling layers, followed by a fully connected classification head.

The classification problem was built by selecting a subset of identities from CelebA, specifically a closed set of identities with sufficient images per class. Images from those identities were split into training and validation subsets so that the label space was consistent across both partitions.

Initial training attempts resulted in near-random performance, approximately 2% accuracy, which corresponds to chance performance over many classes. This problem was corrected by adding proper input normalization, reducing the learning rate, increasing the number of training epochs, and introducing light data augmentation such as horizontal flips and small rotations.

After these fixes, the model successfully converged. The final performance of the from-scratch CNN was:

Training accuracy: approximately 90–91%
Best validation accuracy: 68.32%

This is a substantial improvement over the earlier failed training runs, but it still reflects a significant generalization gap.

**4.3 Comparison of Training Approaches**

A direct comparison between the transfer-learning classifier and the from-scratch CNN shows a clear difference in generalization:

Model	Training Accuracy	Validation Accuracy
Transfer Learning	~94–95%	~92–93%
From-Scratch CNN	~90-91%	68.32%

This comparison highlights the importance of large-scale pretraining for face recognition tasks. The pretrained backbone already encodes rich facial representations learned from a much larger and more diverse dataset, while the from-scratch model must learn low-level and high-level facial features directly from a comparatively small closed-set subset of CelebA.

As a result, the from-scratch model is much more prone to overfitting and does not generalize nearly as well as the transfer-learning model.

### 5. Analysis and Discussion

The pretrained verification model achieves strong generalization performance, with a training verification accuracy of 93.55% and a validation accuracy of 93.20%. The small gap between training and validation performance indicates that the threshold selection and evaluation procedure generalize well across unseen identities, and that the identity-disjoint split successfully prevents overfitting.

Under localized occlusion, validation accuracy decreases to 90.30%, corresponding to a drop of approximately 2.9 percentage points. This indicates that while the embedding model is relatively robust, it is still sensitive to structured local perturbations.

A more detailed analysis of embedding distances reveals that the perturbation primarily affects same-identity pairs. The mean cosine distance for same-identity pairs increases from 0.403 to 0.460, indicating that images of the same individual become less similar under occlusion. In contrast, the mean distance for different-identity pairs remains largely unchanged (0.960 vs. 0.956).

This suggests that localized perturbations primarily disrupt within-class consistency rather than between-class separation. In other words, the model remains capable of distinguishing different individuals, but becomes less reliable at recognizing multiple images of the same person as belonging to the same identity. This observation aligns well with the idea that face-recognition systems may be particularly vulnerable in preserving fine-grained identity consistency under localized perturbations.

The classification experiments provide a complementary perspective. The transfer-learning classifier shows that strong facial representations can support highly accurate identity recognition with relatively little task-specific training. By contrast, the from-scratch CNN achieves much lower validation accuracy and exhibits a much larger train/validation gap, demonstrating how difficult it is to learn robust face representations directly from limited data.

This comparison is one of the most important findings of the project. It shows that the success of face-recognition systems depends not only on architecture, but also on prior learned representations. Large-scale pretraining plays a critical role in enabling good generalization.

Furthermore, the strong dependence on image resolution highlights the role of spatial frequency in face recognition. At lower resolution (112×112), both clean and perturbed performance degrade substantially, indicating that important identity features are lost. At higher resolution (224×224), performance improves, suggesting that higher-frequency features contribute significantly to embedding quality. This also implies that perturbations targeting certain spatial frequencies could become more effective depending on image scale.


### 6. Limitations and Future Work

While the current experiments provide meaningful insight into both learning and robustness, several limitations remain.

First, the perturbation used in the robustness experiments is a simple localized square occlusion rather than a fully optimized adversarial patch. This provides a controlled baseline but does not yet represent the strongest possible attack. A natural extension would be to evaluate gradient-based perturbations such as FGSM or PGD, and eventually compare them to structured patch-based attacks.

Second, the verification experiments use a single pretrained embedding model. Future work could evaluate whether the same robustness trends hold across multiple architectures.

Third, the from-scratch classifier is trained on a limited closed-set subset of CelebA. Although this is sufficient to demonstrate learning and generalization behavior, larger and more balanced datasets would likely improve performance.

Fourth, patch placement is currently random and not tied to specific facial landmarks. A more detailed study could examine whether perturbations placed near highly informative regions such as the eyes, nose, or mouth cause larger degradations in performance.

Finally, additional frequency-based analysis could strengthen the connection between resolution changes and convolutional sensitivity, especially given the observed differences in robustness across 112×112, 160×160, and 224×224.


****

## Part 5: Test Results and Feature-Specific Generalization Analysis

### 1. Test Database and Experimental Design

For Part 5, I evaluated the final neural-network-based face recognition system on a held-out test condition designed to assess how well the model generalizes when only limited facial information is available. The test database was derived from the closed-set CelebA identity classification setup used in Part 4. The classifier was trained on a subset of CelebA identities and evaluated on held-out images from the same closed-set identities, but the Part 5 test condition differs substantially from the original training and validation conditions.

The original training and validation images are standard face images resized and normalized for input into the convolutional neural network. In contrast, the Part 5 test images are transformed using semantic face parsing. I used a BiSeNet-based face parsing model to segment each face image into semantic regions such as eyes, eyebrows, nose, mouth, hair, and skin. For each test condition, only one semantic region was preserved while the rest of the image was blurred. This creates a feature-specific test set that is different from the training and validation data because the classifier was not trained on images where most facial information was removed.

This test setup is useful for evaluating generalization because it measures whether the classifier can still recognize identity when only a restricted facial feature remains clear. Unlike random occlusion, which removes arbitrary image regions, semantic feature-preserving blur produces interpretable test conditions. Each test condition asks a specific question: how much identity information is retained by the eyes alone, the nose alone, the mouth alone, the hair alone, or the skin region alone?

The test set used for this experiment consisted of 200 held-out validation/test images from the closed-set classification subset. The clean version of these images provides a baseline accuracy, while the feature-specific blurred versions serve as out-of-distribution test cases. Although the images come from the same identity set, the feature-preserved versions are visually and statistically different from the normal training images. Therefore, this experiment tests the model’s ability to generalize beyond ordinary full-face inputs.

### Feature-Specific Test Examples

The figure below shows the original image, the BiSeNet parsing output, and several feature-only blurred test conditions.

<p align="center">
  <img src="figures/part5_original.jpg" width="180"/>
  <img src="figures/part5_parsing_overlay.jpg" width="180"/>
  <img src="figures/part5_eyes_only_blur.jpg" width="180"/>
</p>

<p align="center">
  <b>Original</b> &nbsp;&nbsp;&nbsp;&nbsp;
  <b>BiSeNet Parsing</b> &nbsp;&nbsp;&nbsp;&nbsp;
  <b>Eyes Only</b>
</p>

<p align="center">
  <img src="figures/part5_nose_only_blur.jpg" width="180"/>
  <img src="figures/part5_mouth_only_blur.jpg" width="180"/>
  <img src="figures/part5_skin_only_blur.jpg" width="180"/>
</p>

<p align="center">
  <b>Nose Only</b> &nbsp;&nbsp;&nbsp;&nbsp;
  <b>Mouth Only</b> &nbsp;&nbsp;&nbsp;&nbsp;
  <b>Skin Only</b>
</p>


### 2. Test Accuracy and Metrics

The main metric used in this section is classification accuracy, consistent with the closed-set classification setup from Part 4. Accuracy is computed as the proportion of test images for which the predicted identity label matches the true identity label. This is the appropriate metric for the from-scratch CNN classifier because the model outputs a discrete identity class.

The clean test subset achieved an accuracy of **64.5%**. This is consistent with the validation performance observed in Part 4, where the from-scratch CNN achieved approximately **65.84%** best validation accuracy. This confirms that the clean test subset reflects the same general level of performance as the earlier validation evaluation.

### Feature-Specific Test Accuracy

| Test Condition | Accuracy | Drop from Clean |
|---|---:|---:|
| Clean test subset | **64.5%** | — |
| Eyes only | **30.5%** | **34.0 pts** |
| Eyebrows only | **31.5%** | **33.0 pts** |
| Nose only | **27.5%** | **37.0 pts** |
| Mouth only | **29.0%** | **35.5 pts** |
| Hair only | **41.0%** | **23.5 pts** |
| Skin only | **47.0%** | **17.5 pts** |

The results show that classification performance decreases substantially when the model is forced to rely on only one localized facial feature. The largest accuracy drop occurs for the nose-only condition, where accuracy falls from 64.5% to 27.5%. The eyes-only and mouth-only conditions also perform poorly, achieving 30.5% and 29.0% accuracy, respectively. This suggests that compact internal facial components are not sufficient by themselves for reliable identity recognition.

The best feature-only results occur for skin and hair. Skin-only images achieve 47.0% accuracy, and hair-only images achieve 41.0% accuracy. These results suggest that broader appearance cues preserve more identity information than small isolated features. Skin regions may retain information about face shape, complexion, and broad facial texture, while hair regions may preserve hairstyle and head-outline cues. Although these cues are not sufficient to match clean performance, they preserve more information than individual internal features such as the eyes, nose, or mouth.


### 3. Analysis of Generalization Behavior

The main conclusion from this experiment is that the classifier depends on distributed facial evidence rather than any single localized feature. When the full face is visible, the CNN can combine many sources of information: eyes, nose, mouth, face shape, skin texture, hair, and spatial relationships between components. When only one region is preserved, most of these relationships are destroyed, and the classifier’s accuracy drops sharply.

This is expected because human face recognition and neural face recognition both rely heavily on the configuration of multiple features. The relative arrangement of the eyes, nose, mouth, skin texture, and face outline carries important identity information. By blurring everything except one semantic region, the test images remove much of the global structure that the model learned during training.

The poor performance of eyes-only, nose-only, and mouth-only images shows that individual compact facial components are not enough for this classifier to reliably identify people. These regions may contain useful information, but not enough on their own. In particular, the nose-only condition performs worst, suggesting that the model does not treat the nose as a sufficiently distinctive standalone feature. The eyes and mouth perform slightly better but still suffer major accuracy drops.

The relatively stronger performance of skin-only and hair-only images is also important. These results indicate that the model may rely more heavily on broad visual patterns than expected. Skin regions may preserve overall face texture and some shape cues, while hair may preserve hairstyle, color, and silhouette. This is useful but also potentially concerning: if a model relies heavily on hair or broad appearance cues, it may be vulnerable to changes in hairstyle, lighting, makeup, or image conditions.


### 4. Why Test Performance Is Worse

The feature-specific test results are worse than the clean validation results because the feature-only images are out-of-distribution relative to the training data. The model was trained on full face images, where identity information is distributed across the entire face and surrounding appearance. It was not trained to classify images where only one semantic feature remains clear while the rest is blurred.

There are several reasons for the performance drop.

First, the transformation removes global facial structure. The CNN learns filters that respond to spatial patterns across the full image. When most of the image is blurred, these learned spatial relationships are disrupted.

Second, the feature-only images remove contextual relationships between facial components. The identity information in a face is not stored only in isolated regions. Instead, it comes from the combination of multiple features and their relative positions.

Third, the feature masks generated by BiSeNet are sometimes small, especially for features such as eyes and eyebrows. Even when preserved sharply, these regions occupy only a small fraction of the image, leaving the classifier with limited visual evidence.

Fourth, blurring the rest of the image changes the image distribution. The model was trained on natural-looking faces, not heavily blurred images. This distribution shift likely reduces confidence and increases prediction errors.

Finally, the from-scratch CNN has limited generalization ability compared with the pretrained model studied earlier. In Part 4, the from-scratch CNN showed a meaningful gap between training and validation accuracy, indicating overfitting. The Part 5 feature-only experiment makes this limitation more visible by testing the model under a more difficult and unfamiliar input condition.


### 5. Improvements and Future Work

Several improvements could reduce the observed error rates.

First, the model could be trained with feature-specific augmentation. During training, some images could be randomly transformed using semantic blur or partial face masking. This would expose the model to feature-limited inputs and may improve robustness.

Second, the dataset could be expanded. The from-scratch CNN is limited by the relatively small number of images per identity. More images per identity and more variation in pose, lighting, and expression would help the model learn more generalizable features.

Third, transfer learning could be used more directly for the classification task. The pretrained model from Part 4 achieved much stronger performance than the from-scratch CNN, suggesting that large-scale pretraining provides more robust facial representations.

Fourth, feature-specific experiments could be extended to combinations of features. For example, testing “eyes + nose,” “eyes + mouth,” or “skin + hair” would help determine which combinations preserve the most identity information.

Fifth, the feature-specific results could guide adversarial patch placement. If certain facial regions strongly affect recognition, those regions may be good candidates for future patch-based perturbation experiments.


### 6. Conclusion

Part 5 evaluates the trained face classification model under a feature-specific test condition using BiSeNet face parsing. The clean test accuracy is 64.5%, while feature-only accuracies drop substantially. Eyes-only, nose-only, and mouth-only conditions perform near 27–31%, while hair-only and skin-only conditions perform better at 41.0% and 47.0%.

These results show that the model does not recognize identity reliably from a single compact facial feature. Instead, it depends on distributed facial evidence and broader appearance cues. The feature-specific test set therefore provides a meaningful generalization challenge and reveals how the model’s performance changes when normal full-face structure is removed.

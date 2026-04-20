# Part 4: Final Solution

## Final System Overview

For the final stage of the project, I evaluated a face-recognition embedding model using a verification-based protocol rather than a closed-set identity classification protocol. This choice was motivated by the structure of modern face-recognition systems, which typically operate by comparing embedding similarity rather than predicting one of a fixed set of identities.

The final system uses:
- a pretrained frozen face-embedding network (`InceptionResnetV1`),
- identity-disjoint train and validation partitions of CelebA,
- pair-based evaluation on same-identity and different-identity image pairs,
- localized benign perturbations to assess robustness,
- and a small image-resolution sweep to test whether perturbation sensitivity changes with scale.

## Why These Metrics Are Appropriate

Because the project focuses on face embeddings, the most appropriate evaluation metrics are:
- **verification accuracy**: whether same-identity pairs are correctly predicted as same and different-identity pairs are correctly predicted as different;
- **mean cosine distance for same-identity pairs**;
- **mean cosine distance for different-identity pairs**.

These metrics are more suitable than plain multiclass classification accuracy because the model is used as a similarity system rather than a closed-set classifier. Verification accuracy provides a threshold-based notion of correctness, while distance summaries explain how well the embedding space separates same and different identities.

## Train and Validation Results

### Clean Verification

Insert your measured values here:

- Train verification accuracy: **X.XX%**
- Validation verification accuracy: **Y.YY%**
- Mean train positive-pair distance: **...**
- Mean train negative-pair distance: **...**
- Mean validation positive-pair distance: **...**
- Mean validation negative-pair distance: **...**

### Robustness Under Localized Perturbation

- Validation clean accuracy: **...**
- Validation perturbed accuracy: **...**
- Clean mean positive-pair distance: **...**
- Clean mean negative-pair distance: **...**
- Perturbed mean positive-pair distance: **...**
- Perturbed mean negative-pair distance: **...**

### Resolution Study

Summarize how the results changed across:
- 112×112
- 160×160
- 224×224

This addresses the idea that perturbation effectiveness may depend on the relationship between input spatial frequencies and the fixed convolutional filters.

## Commentary on Observed Accuracy

Discuss:
- whether training accuracy is much higher than validation accuracy,
- whether the threshold generalizes well across unseen identities,
- whether localized perturbation causes a substantial drop in validation verification accuracy,
- whether the distance gap between same and different pairs narrows under perturbation.

If training performance is much better than validation performance, that suggests overfitting in the evaluation protocol or threshold selection, even though the backbone itself is frozen. It would indicate that the current setup generalizes less effectively across unseen identities.

If clean accuracy is high but perturbed accuracy drops significantly, that suggests the embedding model is accurate under normal conditions but brittle under localized interference. That would support the core motivation of the project.

## Ideas for Improvement

Potential improvements include:
- stronger and more realistic augmentation during evaluation,
- more controlled perturbation placement strategies,
- evaluation on additional datasets such as LFW,
- more extensive threshold selection or cross-validation,
- a larger and more balanced pair set,
- and, if time allowed, comparison to more explicitly adversarial baselines discussed in the literature.

## How to Run the Final Solution

### Clean verification
```bash
python part4/eval_verification.py --image-size 160

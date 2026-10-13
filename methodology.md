# Methodology

1. Discover class names from folders in the selected dataset.
2. Collect supported image paths and integer labels.
3. Create stratified train, validation, and test splits with a fixed random seed.
4. Load batches and resize to 224 × 224 RGB.
5. Use model-embedded rescaling and training-only random augmentation.
6. Train a CNN with Adam and sparse categorical cross-entropy.
7. Monitor validation loss with EarlyStopping and ModelCheckpoint.
8. Evaluate once on the held-out test set; produce accuracy, per-class precision/recall/F1, a confusion matrix, and plots.

## Important caveat
The split is file-level. If a dataset includes duplicate or near-duplicate photographs, audit them before training because related images across partitions can inflate test metrics. Test results reflect only the chosen dataset and split.

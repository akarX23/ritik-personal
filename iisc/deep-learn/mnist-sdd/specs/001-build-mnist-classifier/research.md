# Research: MNIST Digit Classifier Pipeline

## Decisions

- Use PyTorch and torchvision only for model and dataset handling.
- Install torch and torchvision from the PyTorch nightly xpu wheel index to match the requested project setup.
- Use matplotlib for all plots and visual summaries.
- Use the standard library csv module for metric persistence instead of adding a data-frame dependency.
- Store outputs locally as CSV files and plot images in the results directory.
- Require the user to choose the device explicitly for training and testing.

## Rationale

- PyTorch matches the requested architecture and keeps the model implementation compact.
- torchvision provides MNIST and standard dataset transforms without extra project code.
- matplotlib is sufficient for training curves, confusion matrices, and comparison plots.
- The csv module keeps dependencies minimal while still supporting downstream analysis.
- Explicit device selection avoids hidden behavior and matches the clarified requirement.

## Alternatives Considered

- Pandas for CSV handling: rejected to keep the dependency list small.
- Automatic CPU fallback when GPU is unavailable: rejected because the user requires explicit device instruction.
- Additional plotting libraries such as seaborn: rejected because matplotlib is enough for the required visuals.
- A more complex model family such as convolutional layers: deferred because the requested fixed MLP architecture is sufficient for MNIST and simpler to verify.
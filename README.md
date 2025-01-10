# CRLM-instance

Code for ISBI-2025 Conference - INSTANCE-AWARE DEEP LEARNING FOR LIVER METASTASIS SEGMENTATION

## Overview 

This project focuses on investigating the use of instance-aware loss approach in deep segmentation models applied to the detection and segmentation of liver colorectal metastases (CRLM).

## Key Features
- Automated liver and tumor segmentation using pre-trained SwinUNETR models
- Support for two model variants: standard SwinUNETR and instance-aware SwinUNETR
- Post-processing with volume-based filtering and connected component analysis
- Output segmentation masks with 3 classes (0: background, 1: liver, 2: tumors)

## Input Requirements
- CT scans must be provided in NIfTI format (.nii.gz)
- Images should be in abdominal CT window settings
- Our models were trained on scans resampled to 1.5×1.5×2.0 mm³ voxels

## Pre-trained Models
Pre-trained weights for two SwinUNETR variants are available:
- SwinUNETRb: Baseline model trained with standard Dice+CE loss
- SwinUNETR•: Enhanced model incorporating instance-aware loss for improved small lesion detection

Access to model weights is provided through our cloud archive [link to be added].

## Usage
Configure the input/output paths and model selection:
```python
directory = "path/to/input/nifti/files"     # Input directory containing .nii.gz files
output_dir = "path/to/save/segmentations"   # Output directory for segmentation results
weights_path = "path/to/model/weights.pth"  # Path to pre-trained model weights

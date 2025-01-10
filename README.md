# CRLM-instance

Code for ISBI-2025 Conference - INSTANCE-AWARE DEEP LEARNING FOR LIVER METASTASIS SEGMENTATION

![fig](https://github.com/user-attachments/assets/b61e76b4-cd49-4a50-939b-4dbb058d7219)

## Overview 

This project focuses on investigating the use of instance-aware loss approach in deep segmentation models applied to the detection and segmentation of liver colorectal metastases (CRLM).

## Key Features
- Automated liver and tumor segmentation using pre-trained SwinUNETR models
- Support for two model variants: standard SwinUNETR and instance-aware SwinUNETR
- Post-processing with volume-based filtering and connected component analysis
- Output segmentation masks with 3 classes (0: background, 1: liver, 2: tumors)

## Input Requirements
- CT scans must be provided in NIfTI format (.nii.gz)
- Images will be processed in abdominal CT window settings
- Our models were trained on scans resampled to 1.5×1.5×2.0 mm³ voxels

## Pre-trained Models
Pre-trained weights for two SwinUNETR variants are available:
- SwinUNETRb: Baseline model trained with standard Dice+CE loss
- SwinUNETR•: Enhanced model incorporating instance-aware loss for improved small lesion detection

Access to model weights is provided through our cloud archive https://drive.google.com/drive/folders/1II_7cp-it7lI2vnUGDgD33SSSHpaROuV?usp=sharing.

## Usage
Configure the input/output paths and model selection:
```python
directory = ""     # Input directory containing .nii.gz files
output_dir = ""   # Output directory for segmentation results
weights_path = ""  # Path to pre-trained model weights
```

Install the dependencies
```bash
pip install -r requirements.txt
```

Run the inference :
```python
python inference.py
```

The script will:

1. Process all .nii.gz files in the input directory
2. Generate segmentation masks with labeled regions:
 - 0: Background
 - 1: Liver tissue
 - 2: Tumor regions
3. Apply post-processing to filter small components and retain only significant lesions
4. Save the results as NIfTI files in the specified output directory

## Performance Notes
As detailed in our paper, the models demonstrate different strengths:
- SwinUNETRb achieves balanced performance across lesion sizes
- SwinUNETR• shows improved detection of small lesions (<1000mm³)
- Both variants achieve high recall but may generate more false positives compared to traditional approaches (eg, nn-Unet).

Post-processing helps filter out small artifact components with a fixed volume threshold to reduce false positives generation.

## Reference
If you use this code in your research, please cite our paper [citation to be added].

## Contact
marwan.abbas@univ-brest.fr

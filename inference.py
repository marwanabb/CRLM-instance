import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

from monai.apps import download_url
from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd
)

from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import nibabel as nib
import numpy as np
from scipy.ndimage import label, measurements

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, ToTensord, 
    Resized, ToNumpyd, EnsureTyped, ResizeWithPadOrCropd,SaveImage
)


def inference(dir: str, output_dir: str, weights_path: str):
      
    files = sorted(os.listdir(dir))
    volumes = [f for f in files if f.endswith(".nii.gz")]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=(64, 64, 64),
        in_channels=1,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True
    )
    
    # Load the weights saved using DataParallel

    checkpoint = torch.load(weights_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    modified_model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(modified_model_state_dict)


    # Check for multiple GPUs and wrap model in DataParallel if available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids = [ 0, 1, 2, 3])

    model.to(device)
    model.eval()
    
    test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])
    
    for target in volumes:
        
        inference_file = [{'image': os.path.join(test_dir, target),
                     'label': os.path.join(test_dir, target)} ]

   
        val_ds = CacheDataset(data=test_file, transform=test_transforms, cache_num=6, cache_rate=1.0, num_workers=4)

        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        
        
        with torch.no_grad():
            img_name = os.path.split(val_ds[0]["image"].meta["filename_or_obj"])[1]
            img = val_ds[0]["image"]

            val_inputs = torch.unsqueeze(img, 1).cuda()
            
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
            
        prediction = torch.argmax(val_outputs, dim=1).cpu().numpy().astype(np.uint8)[0]

        saver = SaveImage(output_dir=output_dir, output_postfix= 'liv-tum', output_ext=".nii.gz", separate_folder = False)
        
        saver(torch.argmax(val_outputs, dim=1))
        
        torch.cuda.empty_cache()
        
        # Load the NIfTI file
        nifti_image = nib.load(os.path.join(output_dir, target.split('.nii')[0]+'_liv-tum.nii.gz'))

        image_data = nifti_image.get_fdata()

        # Threshold the image data to create a binary image
        binary_data = np.where(image_data > 0, 1, 0)  # Assumes any non-zero value is part of the object

        # Label the connected components
        labeled_data, num_features = label(binary_data)

        # Find the largest connected component
        component_sizes = measurements.sum(binary_data, labeled_data, index=range(num_features + 1))
        largest_component = (component_sizes == component_sizes.max())

        # Create a binary mask of the largest component
        largest_component_mask = largest_component[labeled_data]

        # Multiply with the original image data to keep the original intensities
        result_data = image_data * largest_component_mask

        # Now, let's handle the tumor components (assuming label 2 for tumors)
        tumor_mask = result_data == 2
        labeled_tumor, num_tumor = label(tumor_mask)

        # Calculate sizes of tumor components
        tumor_sizes = measurements.sum(tumor_mask, labeled_tumor, index=range(num_tumor + 1))

        # Get voxel volume in mm³
        voxel_volume = np.prod(nifti_image.header.get_zooms())

        # Create a mask for tumors larger than 50 mm³
        valid_tumors = np.zeros_like(tumor_sizes, dtype=bool)
        valid_tumors[1:] = tumor_sizes[1:] * voxel_volume >= 50  # Exclude background (index 0)

        # Apply the mask to remove small tumor components
        result_data[labeled_tumor > 0] = 2 * valid_tumors[labeled_tumor[labeled_tumor > 0]]

        # Save the resulting image as a new NIfTI file
        new_nifti_image = nib.Nifti1Image(result_data, nifti_image.affine)
        nib.save(new_nifti_image, os.path.join(output_dir, target.split('.nii')[0]+'_liv-tum.nii.gz'))
            

directory = ''
output_dir = ''
weights_folder = ''
    

inference(directory, output_dir, weights_path)

import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from contextlib import contextmanager
from typing import List, Dict, Optional
import nibabel as nib
from scipy.ndimage import label, measurements
from collections import OrderedDict

from monai.apps import download_url
from monai.utils import set_determinism
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import (
   Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
   ScaleIntensityRanged, CropForegroundd, ToTensord, SaveImage
)
from monai.data import (
   DataLoader,
   CacheDataset
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceConfig:
   """Configuration class for inference parameters"""
   def __init__(self):
       self.img_size = (64, 64, 64)
       self.feature_size = 48
       self.in_channels = 1
       self.out_channels = 3
       self.spacing = (1.5, 1.5, 2.0)
       self.intensity_range = {"a_min": -21, "a_max": 189, "b_min": 0.0, "b_max": 1.0}
       self.sliding_window_size = (96, 96, 96)
       self.minimum_tumor_volume = 50  # mm³
       self.batch_size = 1
       self.num_workers = 4
       self.overlap = 0.8

@contextmanager
def inference_context(model: torch.nn.Module, device: torch.device):
   """Context manager for inference"""
   try:
       model.eval()
       torch.set_grad_enabled(False)
       yield
   finally:
       clear_gpu_memory()

def clear_gpu_memory():
   """Clear GPU memory and synchronize CUDA operations"""
   torch.cuda.empty_cache()
   if torch.cuda.is_available():
       torch.cuda.synchronize()

def setup_model(config: InferenceConfig, weights_path: str, device: torch.device) -> torch.nn.Module:
   """Initialize and load the model"""
   model = SwinUNETR(
       img_size=config.img_size,
       in_channels=config.in_channels,
       out_channels=config.out_channels,
       feature_size=config.feature_size,
       drop_rate=0.0,
       attn_drop_rate=0.0,
       dropout_path_rate=0.0,
       use_checkpoint=True
   )
   
   checkpoint = torch.load(weights_path, map_location=device)
   model_state_dict = checkpoint['model_state_dict']
   modified_model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
   model.load_state_dict(modified_model_state_dict)

   if torch.cuda.device_count() > 1:
       logger.info(f"Using {torch.cuda.device_count()} GPUs!")
       model = torch.nn.DataParallel(model)
   
   return model.to(device)

def get_transforms(config: InferenceConfig) -> Compose:
   """Set up data transforms"""
   return Compose([
       LoadImaged(keys=["image", "label"]),
       EnsureChannelFirstd(keys=["image", "label"]),
       Orientationd(keys=["image", "label"], axcodes="RAS"),
       Spacingd(
           keys=["image", "label"],
           pixdim=config.spacing,
           mode=("bilinear", "nearest"),
       ),
       ScaleIntensityRanged(
           keys=["image"], 
           **config.intensity_range,
           clip=True
       ),
       CropForegroundd(keys=["image", "label"], source_key="image"),
       ToTensord(keys=["image", "label"]),
   ])

def post_process_prediction(nifti_path: str, output_path: str, min_volume: float):
   """Post-process prediction with volume filtering"""
   nifti_image = nib.load(nifti_path)
   image_data = nifti_image.get_fdata()
   
   # Process liver segmentation
   binary_data = np.where(image_data > 0, 1, 0)
   labeled_data, num_features = label(binary_data)
   component_sizes = measurements.sum(binary_data, labeled_data, range(num_features + 1))
   largest_component_mask = (component_sizes == component_sizes.max())[labeled_data]
   result_data = image_data * largest_component_mask

   # Process tumor components
   tumor_mask = result_data == 2
   labeled_tumor, num_tumor = label(tumor_mask)
   tumor_sizes = measurements.sum(tumor_mask, labeled_tumor, range(num_tumor + 1))
   voxel_volume = np.prod(nifti_image.header.get_zooms())
   
   valid_tumors = np.zeros_like(tumor_sizes, dtype=bool)
   valid_tumors[1:] = tumor_sizes[1:] * voxel_volume >= min_volume
   
   result_data[labeled_tumor > 0] = 2 * valid_tumors[labeled_tumor[labeled_tumor > 0]]
   
   new_nifti_image = nib.Nifti1Image(result_data, nifti_image.affine)
   nib.save(new_nifti_image, output_path)

def inference(dir: str, output_dir: str, weights_path: str):
   """Main inference function"""
   # Input validation
   if not os.path.exists(dir):
       raise ValueError(f"Input directory {dir} does not exist")
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)
   if not os.path.exists(weights_path):
       raise ValueError(f"Weights file {weights_path} does not exist")

   config = InferenceConfig()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Setup model and transforms
   model = setup_model(config, weights_path, device)
   transforms = get_transforms(config)
   
   # Get list of volumes
   volumes = [f for f in sorted(os.listdir(dir)) if f.endswith(".nii.gz")]
   logger.info(f"Found {len(volumes)} volumes to process")
   
   for target in tqdm(volumes, desc="Processing volumes"):
       try:
           inference_file = [{
               'image': os.path.join(dir, target),
               'label': os.path.join(dir, target)
           }]
           
           # Prepare dataset and dataloader
           val_ds = CacheDataset(
               data=inference_file,
               transform=transforms,
               cache_num=6,
               cache_rate=1.0,
               num_workers=config.num_workers
           )
           
           val_loader = DataLoader(
               val_ds,
               batch_size=config.batch_size,
               shuffle=False,
               num_workers=config.num_workers,
               pin_memory=True
           )
           
           with inference_context(model, device):
               img = val_ds[0]["image"]
               val_inputs = torch.unsqueeze(img, 1).cuda()
               
               val_outputs = sliding_window_inference(
                   val_inputs,
                   config.sliding_window_size,
                   4,
                   model,
                   overlap=config.overlap
               )
               
               # Save initial prediction
               saver = SaveImage(
                   output_dir=output_dir,
                   output_postfix='liv-tum',
                   output_ext=".nii.gz",
                   separate_folder=False
               )
               saver(torch.argmax(val_outputs, dim=1))
               
               # Post-process the prediction
               prediction_path = os.path.join(output_dir, f"{target.split('.nii')[0]}_liv-tum.nii.gz")
               post_process_prediction(
                   prediction_path,
                   prediction_path,
                   config.minimum_tumor_volume
               )
               
               logger.info(f"Successfully processed {target}")
               
       except Exception as e:
           logger.error(f"Error processing {target}: {str(e)}")
           continue
            
if __name__ == "__main__":
   directory = ""  # Input directory path
   output_dir = ""  # Output directory path
   weights_path = ""  # Model weights path
   
   inference(directory, output_dir, weights_path)

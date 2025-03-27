import os
import numpy as np
import pandas as pd
import torch
import glob
import rasterio
from datetime import datetime, timedelta
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision.transforms import transforms
from tqdm import tqdm
import cv2

class WildfireDataset(Dataset):
    """Dataset for wildfire patch data stored in TIF files."""
    def __init__(
        self,
        file_paths=None,
        root_dir=None,
        years=None,
        variables=None,
        patch_size=(256, 256),
        missing_regions_path=None,
        polygon_coords_path=None,
        transform=None,
        partition='train',
    ):
        super().__init__()
        self.variables = variables
        self.patch_size = patch_size
        self.transform = transform  
        self.partition = partition 
        
        # Load missing regions data
        self.missing_regions = None
        if missing_regions_path:
            self.missing_regions = pd.read_csv(missing_regions_path)
        
        # Load polygon coordinates
        self.polygon_coords = {}
        if polygon_coords_path:
            with open(polygon_coords_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        polygon_id = int(parts[0].split(' ')[1])
                        coords_str = parts[1].strip()
                        # Parse the coordinates string into a list of coordinates
                        try:
                            coords_str = coords_str.replace('[[[', '').replace(']]]', '')
                            coord_pairs = coords_str.split('], [')
                            coords = []
                            for pair in coord_pairs:
                                pair = pair.replace('[', '').replace(']', '')
                                x, y = map(float, pair.split(', '))
                                coords.append((x, y))
                            self.polygon_coords[polygon_id] = coords
                        except Exception as e:
                            print(f"Error parsing coordinates for polygon {polygon_id}: {e}")
        
        # Use provided file paths or build file list
        if file_paths is not None:
            self.file_paths = file_paths
        else:
            self.file_paths = []
            self._build_file_list(root_dir, years)
        

        
    def _build_file_list(self, root_dir, years):
        """Build list of all TIF files to process."""
        for year in years:
            year_dir = os.path.join(root_dir, str(year))
            if not os.path.exists(year_dir):
                continue
                
            # Get all TIF files for this year
            tif_pattern = os.path.join(year_dir, "*.tif")
            year_files = glob.glob(tif_pattern)
            self.file_paths.extend(year_files)
        
    def _get_missing_channels(self, date_str, patch_id):
        """Get list of missing channels for a specific date and patch."""
        if self.missing_regions is None:
            return []
            
        # Filter by date and region
        date_region_filter = (self.missing_regions['date'] == date_str) & (self.missing_regions['region'] == patch_id)
        if date_region_filter.any():
            row = self.missing_regions[date_region_filter].iloc[0]
            # Get the variable names where True indicates the variable is missing
            missing_vars = [col for col in row.index[2:] if row[col]]
            return missing_vars
        return []
        
    def _parse_filename(self, filepath):
        """Extract date and patch ID from filename."""
        filename = os.path.basename(filepath)
        # Assuming filename format: YYYY-MM-DD_patch_id.tif
        parts = filename.split('_')
        date_str = parts[0]
        patch_id = int(parts[1].split('.')[0])
        return date_str, patch_id
        

    def _pad_or_crop(self, data, target_size, resize=False):
        """
        Pad or crop data to target size. If resize=True, uses OpenCV (cv2)
        for potentially faster CPU resizing, falling back to PyTorch's interpolate
        if cv2 fails, and finally to pad/crop if resizing is disabled or fails.

        Args:
            data (np.ndarray): Input data array with shape (H, W, C).
            target_size (tuple[int, int]): Target size as (height, width).
            resize (bool): If True, resize using interpolation. If False, pad/crop only.

        Returns:
            np.ndarray: Processed data array with shape (target_height, target_width, C).
        """
        h, w, c = data.shape
        th, tw = target_size # target_height, target_width

        # --- Resize using OpenCV (cv2) if enabled ---
        if resize:
            # Check if resizing is actually needed
            if h == th and w == tw:
                return data # Already correct size

            try:
                # Note: cv2.resize expects target size as (width, height)
                # Use cv2.INTER_AREA for downsampling (generally recommended)
                # or cv2.INTER_LINEAR for bilinear (closer to original PyTorch code)
                interpolation_method = cv2.INTER_AREA if (th < h or tw < w) else cv2.INTER_LINEAR

                resized_data = cv2.resize(
                    data,
                    (tw, th), # Target width, target height order for cv2
                    interpolation=interpolation_method
                )
                return resized_data

            except ImportError:
                 print("WARNING: cv2 not found. Falling back to PyTorch interpolate for resizing.")
                 resize = False # Disable resize for subsequent fallback logic
            except Exception as e:
                print(f"WARNING: cv2.resize failed ({e}), falling back to PyTorch interpolate.")
                # --- PyTorch Fallback ---
                try:
                    # Ensure data is float for PyTorch interpolate if not already
                    is_float = np.issubdtype(data.dtype, np.floating)
                    tensor_data = torch.from_numpy(data).permute(2, 0, 1) # (C, H, W)
                    if not is_float:
                        tensor_data = tensor_data.float()

                    resized_tensor = torch.nn.functional.interpolate(
                        tensor_data.unsqueeze(0), # Add batch dim
                        size=(th, tw),            # Target height, width
                        mode='bilinear',          # Or 'area' if preferred for downsampling
                        align_corners=False       # Generally recommended to be False
                        # Use antialias=True for potentially better quality downsampling in newer PyTorch versions
                        # antialias=True if (th < h or tw < w) else False
                    ).squeeze(0) # Remove batch dim

                    # Convert back to numpy and original dtype
                    resized_data_torch = resized_tensor.permute(1, 2, 0).numpy()
                    if resized_data_torch.dtype != data.dtype:
                         resized_data_torch = resized_data_torch.astype(data.dtype)

                    # print(f"Debug: Resized {h}x{w} -> {resized_data_torch.shape[0]}x{resized_data_torch.shape[1]} using PyTorch fallback")
                    return resized_data_torch

                except Exception as e_torch:
                     print(f"ERROR: PyTorch fallback resize also failed ({e_torch}). Reverting to pad/crop.")
                     resize = False # Disable resize, proceed to pad/crop logic below

        # --- Pad/Crop Logic (if resize=False or resize attempts failed) ---
        # Crop first if larger than target
        if h > th:
            data = data[:th, :, :]
            h = th # Update height
        if w > tw:
            data = data[:, :tw, :]
            w = tw # Update width

        # Pad if smaller than target
        pad_h = th - h
        pad_w = tw - w

        if pad_h > 0 or pad_w > 0:
            # Calculate padding widths: ((top, bottom), (left, right), (before_c, after_c))
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            # Using np.pad is generally clean for zero-padding
            try:
                padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)
                # print(f"Debug: Padded {h}x{w} -> {padded_data.shape[0]}x{padded_data.shape[1]}")
                return padded_data
            except Exception as e_pad:
                 print(f"ERROR: np.pad failed ({e_pad}). Returning original cropped data.")
                 # Fallback if even padding fails (unlikely)
                 return data
        else:
             # print(f"Debug: Data already target size {h}x{w}. No pad/crop needed.")
             return data # Return data if it was already the correct size after potential cropping
    
    
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        date_str, patch_id = self._parse_filename(filepath)
        
        # Read TIF file using rasterio
        with rasterio.open(filepath) as src:
            # Read all bands
            data = src.read()
            # Transpose from (C, H, W) to (H, W, C)
            data = data.transpose(1, 2, 0)
        
        # Pad or crop to target size
        data = self._pad_or_crop(data, self.patch_size)
        
        # Get missing channels for this date and patch
        missing_vars = self._get_missing_channels(date_str, patch_id)
        
        # Create a tensor of zeros with the full expected size
        h, w = data.shape[:2]
        full_data = np.zeros((h, w, len(self.variables)), dtype=data.dtype)
        
        # Map the loaded data to the correct channel positions
        current_channel = 0
        for i, var in enumerate(self.variables):
            if var not in missing_vars and current_channel < data.shape[2]:
                full_data[:, :, i] = data[:, :, current_channel]
                current_channel += 1
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(full_data).float()
        data_tensor = data_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Apply transforms if provided
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, self.variables

class WildfireForecast(Dataset):
    """Forecast dataset for wildfire data that pairs input and future output data."""
    def __init__(
        self, 
        dataset, 
        max_predict_range=7,  # Default to 7 days prediction
        min_predict_range=1,  # Default minimum 1 day ahead 
        random_lead_time=False
    ):
        super().__init__()
        self.dataset = dataset
        self.max_predict_range = max_predict_range
        self.min_predict_range = min_predict_range
        self.random_lead_time = random_lead_time
        
        # Build file pairs for prediction based on dates
        self.input_output_pairs = self._build_forecast_pairs()
        
    def _build_forecast_pairs(self):
        """Build pairs of (input_idx, output_idx) for forecasting.
        
        For each sample, directly samples k lead times and checks if corresponding
        target dates exist, avoiding iteration through all dates.
        """
        # Number of forecast pairs to attempt per input sample
        k_samples = 3 if self.random_lead_time else 1
        
        # Create a mapping from date+patch to file index
        date_patch_to_idx = {}
        
        # Build indices
        for idx, filepath in tqdm(enumerate(self.dataset.file_paths), total=len(self.dataset.file_paths), desc="Building forecast pairs"):
            date_str, patch_id = self.dataset._parse_filename(filepath)
            date_patch_to_idx[(date_str, patch_id)] = idx
        
        # Store all valid input_idx, output_idx pairs
        valid_pairs = []
        
        # Process each input sample
        for input_idx, filepath in tqdm(enumerate(self.dataset.file_paths), total=len(self.dataset.file_paths), desc="Building forecast pairs"):
            input_date_str, patch_id = self.dataset._parse_filename(filepath)
            input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
            
            if self.random_lead_time:
                # Generate possible lead times based on our parameters
                possible_lead_times = list(range(self.min_predict_range, self.max_predict_range + 1))
                
                # Sample k lead times (or use all if fewer than k)
                if len(possible_lead_times) > k_samples:
                    sampled_lead_times = np.random.choice(
                        possible_lead_times,
                        size=k_samples,
                        replace=False
                    )
                else:
                    # If not enough options, use all available lead times up to k_samples
                    sampled_lead_times = possible_lead_times[:k_samples]
            else:
                # If not using random lead time, always use exactly max_predict_range
                sampled_lead_times = [self.max_predict_range]
            
            # For each sampled lead time, directly check if target date exists
            for lead_time in sampled_lead_times:
                lead_time = int(lead_time)
                target_date = input_date + timedelta(days=lead_time)
                target_date_str = target_date.strftime("%Y-%m-%d")
                
                # Check if the target date exists for this patch
                if (target_date_str, patch_id) in date_patch_to_idx:
                    output_idx = date_patch_to_idx[(target_date_str, patch_id)]
                    valid_pairs.append((input_idx, output_idx, lead_time))
        
        return valid_pairs
    
    def __len__(self):
        return len(self.input_output_pairs)
    
    def __getitem__(self, idx):
        input_idx, output_idx, days_diff = self.input_output_pairs[idx]
        
        # Get input data
        input_data, input_variables = self.dataset[input_idx]
        
        # Get output data
        output_data, output_variables = self.dataset[output_idx]
        lead_time = torch.tensor(days_diff, dtype=torch.float32)
        
        return input_data, output_data, lead_time, input_variables, output_variables

class WildfireDataModule(LightningDataModule):
    """
    DataModule for wildfire patch data, handling data loading, splitting,
    normalization, and forecast pair creation. Allows using pre-calculated
    normalization statistics.
    """
    def __init__(
        self,
        root_dir,
        variables,
        means=None,  # <-- Accept pre-calculated means
        stds=None,   # <-- Accept pre-calculated stds
        missing_regions_path=None,
        polygon_coords_path=None,
        patch_size=(64, 64),
        prediction_range=7,  # Maximum days to predict ahead
        min_prediction_range=1,  # Minimum days to predict ahead
        random_lead_time=False,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    ):
        """
        Args:
            root_dir (str): Path to the root directory containing year subdirectories.
            variables (list[str]): List of variable names corresponding to TIF channels.
            means (list[float], optional): Pre-calculated mean for each variable. Defaults to None.
            stds (list[float], optional): Pre-calculated standard deviation for each variable. Defaults to None.
            missing_regions_path (str, optional): Path to the CSV listing missing channels. Defaults to None.
            polygon_coords_path (str, optional): Path to the TXT file with polygon coordinates. Defaults to None.
            patch_size (tuple[int, int], optional): Target size (height, width) for patches. Defaults to (64, 64).
            prediction_range (int, optional): Maximum days ahead for forecast targets. Defaults to 7.
            min_prediction_range (int, optional): Minimum days ahead for forecast targets. Defaults to 1.
            random_lead_time (bool, optional): If True, sample lead times randomly between min and max range.
                                               If False, use fixed max_prediction_range. Defaults to False.
            val_ratio (float, optional): Fraction of data for the validation set. Defaults to 0.15.
            test_ratio (float, optional): Fraction of data for the test set. Defaults to 0.15.
            random_seed (int, optional): Seed for shuffling data splits. Defaults to 42.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
            pin_memory (bool, optional): If True, copies Tensors into CUDA pinned memory before returning them. Defaults to True.
        """
        super().__init__()
        # Save hyperparameters, including the new means and stds
        # logger=False prevents these from cluttering tensorboard hparams tab
        self.save_hyperparameters(logger=False)

        # --- Validation for provided means and stds ---
        if self.hparams.means is not None or self.hparams.stds is not None:
            if self.hparams.means is None or self.hparams.stds is None:
                raise ValueError("Both 'means' and 'stds' must be provided together, or both left as None.")
            if len(self.hparams.means) != len(self.hparams.variables):
                raise ValueError(f"Length of 'means' ({len(self.hparams.means)}) must match length of 'variables' ({len(self.hparams.variables)}).")
            if len(self.hparams.stds) != len(self.hparams.variables):
                raise ValueError(f"Length of 'stds' ({len(self.hparams.stds)}) must match length of 'variables' ({len(self.hparams.variables)}).")
            print("INFO: Using provided means and stds for normalization.")
        else:
            print("INFO: Means and stds not provided, will calculate from a sample of training data.")
        # --- End Validation ---

        # Set default paths if not provided and check existence
        if self.hparams.missing_regions_path is None:
            potential_path = os.path.join(self.hparams.root_dir, "missing_channels.csv")
            if os.path.exists(potential_path):
                self.hparams.missing_regions_path = potential_path
                print(f"INFO: Using missing_channels.csv from data directory: {self.hparams.missing_regions_path}")
            else:
                print(f"WARNING: missing_regions_path not provided and not found at default location: {potential_path}")
                # Keep it None if not found

        if self.hparams.polygon_coords_path is None:
            potential_path = os.path.join(self.hparams.root_dir, "polygon_coords.txt")
            if os.path.exists(potential_path):
                self.hparams.polygon_coords_path = potential_path
                print(f"INFO: Using polygon_coords.txt from data directory: {self.hparams.polygon_coords_path}")
            else:
                print(f"WARNING: polygon_coords_path not provided and not found at default location: {potential_path}")
                # Keep it None if not found

        # Discover available years in the data directory
        self.available_years = self._discover_years(self.hparams.root_dir)
        if not self.available_years:
            print(f"WARNING: No year subdirectories found in root_dir: {self.hparams.root_dir}")

        # Initialize dataset attributes
        self.transforms = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.all_files = [] # Store all found file paths


    def _discover_years(self, root_dir):
        """Discover available year subdirectories in the data root directory."""
        available_years = []
        try:
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    available_years.append(int(item))
        except FileNotFoundError:
             print(f"ERROR: root_dir not found: {root_dir}")
             return []
        except Exception as e:
             print(f"ERROR: Could not list directories in root_dir {root_dir}: {e}")
             return []

        available_years.sort()
        return available_years

    def setup(self, stage=None):
        """
        Set up the datasets for train, validation, and test stages.
        This includes finding files, splitting, creating normalization transform,
        and instantiating Dataset objects.

        Args:
            stage (str, optional): Either 'fit', 'validate', 'test', or 'predict'.
                                   Used to setup specific datasets. Defaults to None.
        """
        # Prevent redundant setup
        if self.all_files and (self.dataset_train or self.dataset_test):
             print("INFO: Datasets already set up.")
             return

        # --- 1. Collect all TIF files ---
        print(f"INFO: Discovering TIF files in {self.hparams.root_dir} for years: {self.available_years}")
        self.all_files = []
        for year in self.available_years:
            year_dir = os.path.join(self.hparams.root_dir, str(year))
            if not os.path.exists(year_dir):
                print(f"WARNING: Directory not found for year {year}: {year_dir}")
                continue

            tif_pattern = os.path.join(year_dir, "*.tif")
            try:
                 year_files = glob.glob(tif_pattern)
                 self.all_files.extend(year_files)
            except Exception as e:
                 print(f"ERROR: Could not glob files in {year_dir}: {e}")

        if not self.all_files:
             raise RuntimeError(f"ERROR: No TIF files found in {self.hparams.root_dir} for the specified years. Check data paths.")
        print(f"INFO: Found {len(self.all_files)} total TIF files.")

        # --- 2. Shuffle and Split Files ---
        np.random.seed(self.hparams.random_seed)
        np.random.shuffle(self.all_files)

        val_size = int(len(self.all_files) * self.hparams.val_ratio)
        test_size = int(len(self.all_files) * self.hparams.test_ratio)
        train_size = len(self.all_files) - val_size - test_size

        if train_size <= 0 or val_size <=0 or test_size <=0:
             print(f"WARNING: Dataset split resulted in zero samples for one or more sets. "
                   f"Train: {train_size}, Val: {val_size}, Test: {test_size}. Check ratios and total file count.")

        train_files = self.all_files[:train_size]
        val_files = self.all_files[train_size : train_size + val_size]
        test_files = self.all_files[train_size + val_size :]

        print(f"INFO: Splitting data: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}")

        # --- 3. Setup Normalization Transform ---
        # Pass train_files ONLY if needed for calculation (i.e., means/stds not provided)
        self.transforms = self._get_normalize(train_files if self.hparams.means is None else None)

        # --- 4. Create Datasets based on stage ---
        if stage == 'fit' or stage is None:
            print("INFO: Setting up Train and Validation datasets...")
            base_train_dataset = WildfireDataset(
                file_paths=train_files,
                variables=self.hparams.variables,
                patch_size=self.hparams.patch_size,
                missing_regions_path=self.hparams.missing_regions_path,
                polygon_coords_path=self.hparams.polygon_coords_path,
                transform=self.transforms,
                partition='train'
            )
            base_val_dataset = WildfireDataset(
                file_paths=val_files,
                variables=self.hparams.variables,
                patch_size=self.hparams.patch_size,
                missing_regions_path=self.hparams.missing_regions_path,
                polygon_coords_path=self.hparams.polygon_coords_path,
                transform=self.transforms,
                partition='val'
            )

            self.dataset_train = WildfireForecast(
                dataset=base_train_dataset,
                max_predict_range=self.hparams.prediction_range,
                min_predict_range=self.hparams.min_prediction_range,
                random_lead_time=self.hparams.random_lead_time
            )
            self.dataset_val = WildfireForecast(
                dataset=base_val_dataset,
                max_predict_range=self.hparams.prediction_range,
                min_predict_range=self.hparams.min_prediction_range,
                # Usually False for validation/test to be consistent
                random_lead_time=False # Or self.hparams.random_lead_time if desired
            )
            print(f"INFO: Train forecast pairs: {len(self.dataset_train)}")
            print(f"INFO: Validation forecast pairs: {len(self.dataset_val)}")


        if stage == 'test' or stage is None:
            print("INFO: Setting up Test dataset...")
            base_test_dataset = WildfireDataset(
                file_paths=test_files,
                variables=self.hparams.variables,
                patch_size=self.hparams.patch_size,
                missing_regions_path=self.hparams.missing_regions_path,
                polygon_coords_path=self.hparams.polygon_coords_path,
                transform=self.transforms,
                partition='test'
            )
            self.dataset_test = WildfireForecast(
                dataset=base_test_dataset,
                max_predict_range=self.hparams.prediction_range,
                min_predict_range=self.hparams.min_prediction_range,
                # Usually False for validation/test to be consistent
                random_lead_time=False # Or self.hparams.random_lead_time if desired
            )
            print(f"INFO: Test forecast pairs: {len(self.dataset_test)}")

        # If stage is 'predict', it often uses the test set or a specific prediction set
        if stage == 'predict':
            # Typically, prediction uses the test set or a dedicated prediction set
            # Re-using test set setup logic here if appropriate
            if self.dataset_test is None:
                 print("INFO: Setting up Test dataset for prediction...")
                 # (Repeat test dataset setup logic if needed, or load a specific predict set)
                 base_test_dataset = WildfireDataset(...)
                 self.dataset_test = WildfireForecast(...)
                 print(f"INFO: Prediction dataset (using test set) pairs: {len(self.dataset_test)}")


    def _get_normalize(self, train_files=None):
        """
        Creates the normalization transform.
        Uses pre-calculated means/stds from hparams if available.
        Otherwise, calculates statistics from a sample of the training files.

        Args:
            train_files (list[str], optional): List of training file paths.
                                               Needed only if means/stds are not provided.

        Returns:
            torchvision.transforms.Normalize: The normalization transform.
        """
        # --- Use provided means and stds if available ---
        if self.hparams.means is not None and self.hparams.stds is not None:
            print("INFO: Creating normalization transform using provided statistics.")
            try:
                mean_tensor = torch.tensor(self.hparams.means, dtype=torch.float32)
                std_tensor = torch.tensor(self.hparams.stds, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"ERROR: Could not convert provided means/stds to tensors: {e}")

            # Ensure no division by zero in std
            zero_std_mask = (std_tensor == 0)
            if zero_std_mask.any():
                 print(f"WARNING: Found {zero_std_mask.sum()} zero values in provided standard deviations. Replacing with 1.0.")
                 std_tensor[zero_std_mask] = 1.0

            return transforms.Normalize(mean_tensor, std_tensor)

        # --- Otherwise, calculate from training data sample ---
        else:
            print("INFO: Calculating normalization statistics from a sample of training data...")
            if not train_files:
                 print("WARNING: Cannot calculate statistics: No training files provided and no pre-calculated stats available. Using default normalization (mean=0, std=1).")
                 return transforms.Normalize(
                    mean=[0.0] * len(self.hparams.variables),
                    std=[1.0] * len(self.hparams.variables)
                 )

            # Use a subset of files for calculating statistics
            sample_size = min(1000, len(train_files)) # Limit sample size for speed
            print(f"INFO: Using a random sample of {sample_size} training files for statistics calculation.")
            try:
                # Ensure we don't sample more than available files
                actual_sample_size = min(sample_size, len(train_files))
                sample_files = np.random.choice(train_files, actual_sample_size, replace=False)
            except Exception as e:
                 print(f"ERROR: Could not sample training files: {e}. Using default normalization.")
                 return transforms.Normalize(
                    mean=[0.0] * len(self.hparams.variables),
                    std=[1.0] * len(self.hparams.variables)
                 )

            # Create a temporary dataset *without* the normalization transform for calculation
            temp_dataset = WildfireDataset(
                file_paths=sample_files.tolist(),
                variables=self.hparams.variables,
                patch_size=self.hparams.patch_size,
                missing_regions_path=self.hparams.missing_regions_path,
                polygon_coords_path=self.hparams.polygon_coords_path,
                transform=None, # Important: No transform during calculation
                partition='train'
            )

            # Use DataLoader for potentially faster iteration and memory management
            # Use batch_size from hparams, but adjust num_workers if needed for temp calc
            temp_loader = DataLoader(
                temp_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=os.cpu_count()//self.hparams.num_workers, # Use fewer workers for temp calc
                pin_memory=False # No need to pin memory for calculation
            )

            # Welford's online algorithm for stable mean/variance calculation is robust,
            # but simple sum/sum_sq is often sufficient and easier here.
            channel_sum = torch.zeros(len(self.hparams.variables), dtype=torch.float64)
            channel_sum_sq = torch.zeros(len(self.hparams.variables), dtype=torch.float64)
            pixel_count = 0
            num_patches = 0

            pbar = tqdm(temp_loader, desc="Calculating Stats", leave=False, total=len(temp_loader))
            try:
                for data, _ in pbar:
                    # data shape: (B, C, H, W)
                    # Ensure data is float64 for precision in sums
                    data_f64 = data.to(torch.float64)
                    # Sum over batch, height, width dimensions
                    channel_sum += data_f64.sum(dim=(0, 2, 3))
                    channel_sum_sq += (data_f64**2).sum(dim=(0, 2, 3))
                    # Count total number of pixels per channel = B * H * W
                    pixel_count += data.size(0) * data.size(2) * data.size(3)
                    num_patches += data.size(0)
            except Exception as e:
                 pbar.close()
                 print(f"ERROR: Failed during statistics calculation: {e}. Using default normalization.")
                 return transforms.Normalize(
                    mean=[0.0] * len(self.hparams.variables),
                    std=[1.0] * len(self.hparams.variables)
                 )
            finally:
                 pbar.close()


            if pixel_count > 0:
                mean = (channel_sum / pixel_count).to(torch.float32)
                # Variance = E[X^2] - (E[X])^2
                variance = (channel_sum_sq / pixel_count) - (mean.to(torch.float64)**2)
                # Clamp variance to avoid negative values due to floating point errors before sqrt
                variance = torch.clamp(variance, min=1e-7)
                std = torch.sqrt(variance).to(torch.float32)

                # Replace near-zero std to avoid division by zero
                zero_std_mask = (std < 1e-6)
                if zero_std_mask.any():
                    print(f"WARNING: Calculated {zero_std_mask.sum()} near-zero standard deviations. Replacing with 1.0.")
                    std[zero_std_mask] = 1.0

                print(f"INFO: Calculated statistics from {num_patches} patches ({pixel_count} pixels per channel):")
                # Limit print precision for readability
                with np.printoptions(precision=6, suppress=True):
                    print(f"  Calculated Mean: {mean.numpy()}")
                    print(f"  Calculated Std:  {std.numpy()}")
                return transforms.Normalize(mean, std)
            else:
                print("WARNING: No data processed during statistics calculation (pixel_count=0). Using default normalization.")
                return transforms.Normalize(
                    mean=[0.0] * len(self.hparams.variables),
                    std=[1.0] * len(self.hparams.variables)
                )

    def get_lat_lon(self):
        """
        Return latitude and longitude arrays (simplified placeholder).
        Requires polygon_coords.txt to be processed correctly by WildfireDataset.
        """
        # Attempt to get coords from a dataset instance if setup
        dataset_instance = self.dataset_train or self.dataset_val or self.dataset_test
        if dataset_instance and hasattr(dataset_instance.dataset, 'polygon_coords') and dataset_instance.dataset.polygon_coords:
            coords_dict = dataset_instance.dataset.polygon_coords
            # This is a simplified approach assuming coords represent grid points
            # A more robust method would use rasterio profile/transform info
            all_lats = set()
            all_lons = set()
            try:
                 for poly_id, coords_list in coords_dict.items():
                      for lon, lat in coords_list:
                           all_lons.add(lon)
                           all_lats.add(lat)

                 if all_lats and all_lons:
                      lats = np.array(sorted(list(all_lats)))
                      lons = np.array(sorted(list(all_lons)))
                      # Check if they form a reasonable grid (optional)
                      if len(lats) * len(lons) > len(all_lats) + len(all_lons) : # Basic check
                           print("INFO: Extracted lat/lon from polygon coordinates.")
                           return lats, lons
                 else:
                      print("WARNING: Polygon coordinates found but could not extract valid lat/lon sets.")

            except Exception as e:
                 print(f"WARNING: Error processing polygon coordinates for lat/lon: {e}")

        # Fallback to default placeholder if coords are unavailable or processing fails
        print("WARNING: Could not get lat/lon from polygon coordinates. Returning default placeholder grid.")
        h, w = self.hparams.patch_size
        # Approximate CONUS range - adjust if your data covers a different region
        lats = np.linspace(24, 50, h)
        lons = np.linspace(-125, -66, w)
        return lats, lons

    def get_climatology(self, partition="val"):
        """
        Calculate the mean climatology for a given data partition (simplified placeholder).

        Args:
            partition (str): "train", "val", or "test".

        Returns:
            torch.Tensor or None: Mean tensor across the specified partition, or None if unavailable.
        """
        print(f"INFO: Attempting to calculate climatology for partition: {partition}")
        dataset = None
        if partition == "val" and self.dataset_val:
            dataset = self.dataset_val
        elif partition == "test" and self.dataset_test:
            dataset = self.dataset_test
        elif partition == "train" and self.dataset_train:
            # Warning: Calculating climatology on the full training set can be very slow!
            print("WARNING: Calculating climatology on the training set can be very time-consuming.")
            dataset = self.dataset_train
        else:
            print(f"WARNING: Dataset for partition '{partition}' is not available.")
            return None

        if len(dataset) == 0:
            print(f"WARNING: Dataset for partition '{partition}' is empty.")
            return None

        # Use a sample for efficiency, especially for large datasets
        # Adjust sample_size as needed
        sample_size = min(500, len(dataset))
        print(f"INFO: Using a sample of {sample_size} items to estimate climatology.")
        indices = np.random.choice(len(dataset), sample_size, replace=False)

        # Use a temporary dataloader for calculation
        temp_loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(indices), # Sample indices
            num_workers=os.cpu_count()//self.hparams.num_workers,
            pin_memory=False
        )

        channel_sum = torch.zeros(len(self.hparams.variables), dtype=torch.float64)
        pixel_count = 0
        num_samples_processed = 0

        pbar = tqdm(temp_loader, desc=f"Calculating Climatology ({partition})", leave=False)
        try:
            # We need the *input* data for climatology usually
            for input_data, _, _, _, _ in pbar:
                data_f64 = input_data.to(torch.float64)
                channel_sum += data_f64.sum(dim=(0, 2, 3))
                pixel_count += input_data.size(0) * input_data.size(2) * input_data.size(3)
                num_samples_processed += input_data.size(0)
        except Exception as e:
            pbar.close()
            print(f"ERROR: Failed during climatology calculation: {e}. Returning None.")
            return None
        finally:
            pbar.close()

        if pixel_count > 0:
            climatology = (channel_sum / pixel_count).to(torch.float32)
            print(f"INFO: Climatology calculated from {num_samples_processed} samples.")
            return climatology
        else:
            print(f"WARNING: No data processed for climatology calculation (pixel_count=0). Returning None.")
            return None


    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        if not self.dataset_train:
            print("ERROR: Training dataset not set up. Call setup('fit') first.")
            return None
        if len(self.dataset_train) == 0:
            print("WARNING: Training dataset is empty!")
            # Return None or an empty loader based on desired behavior
            return None

        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True, # Shuffle training data
            num_workers=os.cpu_count()//self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn, # Assumes collate_fn is defined globally or imported
            persistent_workers=True if self.hparams.num_workers > 0 else False, # Can speed up epoch starts
            drop_last=False # Keep partial batches
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        if not self.dataset_val:
            print("ERROR: Validation dataset not set up. Call setup('fit') or setup() first.")
            return None
        if len(self.dataset_val) == 0:
            print("WARNING: Validation dataset is empty!")
            return None

        return DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=False, # No shuffling for validation
            num_workers=os.cpu_count()//self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            drop_last=False
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        if not self.dataset_test:
            print("ERROR: Test dataset not set up. Call setup('test') or setup() first.")
            return None
        if len(self.dataset_test) == 0:
            print("WARNING: Test dataset is empty!")
            return None

        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            shuffle=False, # No shuffling for test
            num_workers=os.cpu_count()//self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            drop_last=False
        )

    def predict_dataloader(self):
        """Returns the DataLoader for prediction (often uses the test set)."""
        if not self.dataset_test: # Or a dedicated prediction dataset if you implement one
             print("ERROR: Prediction dataset (using test set) not set up. Call setup('predict') or setup() first.")
             return None
        if len(self.dataset_test) == 0:
             print("WARNING: Prediction dataset (using test set) is empty!")
             return None

        # Typically prediction uses the same settings as test
        return DataLoader(
            self.dataset_test, # Or self.dataset_predict if you add it
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()//self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            drop_last=False
        )


def collate_fn(batch):
    """Custom collate function for the wildfire forecast dataloader.
    
    Takes a batch of samples from WildfireForecast and collates them
    into a single batch tensor with proper handling of variable names.
    
    Args:
        batch: List of tuples (input_data, output_data, lead_time, input_vars, output_vars)
    
    Returns:
        Tuple of (batched_inputs, batched_outputs, batched_lead_times, input_vars, output_vars)
    """
    # Check if we have any samples
    if len(batch) == 0:
        return None
    
    # Extract components and stack tensors
    inputs = torch.stack([item[0] for item in batch])
    outputs = torch.stack([item[1] for item in batch])
    lead_times = torch.stack([item[2] for item in batch])
    
    # All samples in a batch should have the same variables
    input_variables = batch[0][3]
    output_variables = batch[0][4]
    
    return inputs, outputs, lead_times, input_variables, output_variables

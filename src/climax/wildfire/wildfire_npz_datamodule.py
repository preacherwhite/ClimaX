import math
import os
import random
import re
from datetime import datetime, timedelta
import csv

import numpy as np
import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
import pickle
import hashlib

from pytorch_lightning import LightningDataModule
class WildfireNpzReader(IterableDataset):
    """Iterable dataset for reading wildfire NPZ files with distributed training support."""
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables=None,
        shuffle=False,
        multi_dataset_training=False,
        valid_dates=None,  # Now contains mapping of dates to valid patch IDs
    ):
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        self.file_list = file_list[start_idx:end_idx]
        self.variables = variables
        self.out_variables = variables  # Always use the same variables for input and output
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training
        self.valid_dates = valid_dates  # Store the mapping of dates to valid patch IDs
        # Caching for variable indices
        self.var_to_idx_cache = {}
        
    def _parse_filename(self, filepath):
        """Extract region and date range from filename."""
        filename = os.path.basename(filepath)
        # Assuming filename format: {region_number}_{start_date}_{end_date}.npz
        match = re.match(r'(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.npz', filename)
        if match:
            region_id = int(match.group(1))
            start_date = match.group(2)
            end_date = match.group(3)
            return region_id, start_date, end_date
        else:
            raise ValueError(f"Invalid filename format: {filename}")
            
    def _map_variables_to_indices(self, npz_columns):
        """Map required variables to column indices in the NPZ file."""
        # Create cache key
        cache_key = tuple(npz_columns)
        
        # Check if mapping is in cache
        if cache_key in self.var_to_idx_cache:
            return self.var_to_idx_cache[cache_key]
        
        # Create mapping from variable name to index
        var_to_idx = {}
        for i, col in enumerate(npz_columns):
            var_to_idx[col] = i
        
        # Cache the mapping
        self.var_to_idx_cache[cache_key] = var_to_idx
        
        return var_to_idx

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
            
        # Distributed training support
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                num_nodes = int(os.environ.get("NODES", 1))
                num_gpus_per_node = int(world_size / num_nodes)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
                
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            filepath = self.file_list[idx]
            try:
                # Load the NPZ file
                npz_data = np.load(filepath, allow_pickle=True)
                
                # Extract dates
                dates = npz_data['dates']
                
                # Get data and columns
                data = npz_data['data']
                columns = npz_data['columns']
                
                # Get region ID from filename
                region_id, _, _ = self._parse_filename(filepath)
                
                # Map variables to column indices
                var_to_idx = self._map_variables_to_indices(columns)
                
                # For each date, yield the data
                for date_idx, date_str in enumerate(dates):
                    # Skip if filtering by valid_dates and either:
                    # 1. The date is not in valid_dates, or
                    # 2. The region_id is not in the set of valid patch IDs for this date
                    if self.valid_dates is not None:
                        if date_str not in self.valid_dates or region_id not in self.valid_dates[date_str]:
                            continue
                        
                    # Get data for this date
                    date_data = data[date_idx]
                    
                    # Process variables
                    processed_data = {}
                    for var in self.variables:
                        if var in var_to_idx:
                            col_idx = var_to_idx[var]
                            var_data = date_data[col_idx]
                            # Handle null or NaN values
                            if var_data is None or np.all(np.isnan(var_data)):
                                # Create zeros array with proper shape
                                shape = date_data[0].shape if date_data[0] is not None else (128, 128)
                                var_data = np.zeros(shape, dtype=np.float32)
                            processed_data[var] = var_data
                            
                    metadata = {
                        'date': date_str,
                        'region_id': region_id,
                        'filepath': filepath
                    }
                    
                    yield processed_data, self.variables, self.variables, metadata
                    
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
                continue


class WildfireForecast(IterableDataset):
    """Creates forecast pairs from wildfire data."""
    def __init__(
        self, 
        dataset, 
        max_predict_range=7,
        min_predict_range=1,
        random_lead_time=False
    ):
        super().__init__()
        self.dataset = dataset
        self.max_predict_range = max_predict_range
        self.min_predict_range = min_predict_range
        self.random_lead_time = random_lead_time

    def _reshape_to_target_size(self, data_array, target_size=(128, 128)):
        """Reshape data to target patch size using random cropping or padding."""
        h, w = data_array.shape[:2]
        th, tw = target_size
        
        # If dimensions match, return as is
        if h == th and w == tw:
            return data_array
        
        # Create output array with target size
        if len(data_array.shape) == 3:
            # For multi-channel data
            c = data_array.shape[2]
            reshaped_data = np.zeros((th, tw, c), dtype=data_array.dtype)
        else:
            # For single-channel data
            reshaped_data = np.zeros((th, tw), dtype=data_array.dtype)
        
        # Case 1: Input is larger than target in both dimensions - random crop
        if h >= th and w >= tw:
            # Choose random starting point for cropping
            start_h = np.random.randint(0, h - th + 1) if h > th else 0
            start_w = np.random.randint(0, w - tw + 1) if w > tw else 0
            
            # Perform the crop
            if len(data_array.shape) == 3:
                reshaped_data = data_array[start_h:start_h+th, start_w:start_w+tw, :]
            else:
                reshaped_data = data_array[start_h:start_h+th, start_w:start_w+tw]
                
        # Case 2: Mixed dimensions - crop where larger, pad where smaller
        else:
            # Determine crop/pad dimensions
            crop_h = min(h, th)
            crop_w = min(w, tw)
            
            # For cropping from source (if needed)
            start_h = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
            start_w = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
            
            # For placement in target (centered with random offset if padding)
            pad_h_before = (th - crop_h) // 2
            pad_w_before = (tw - crop_w) // 2
            
            # Add small random offset to padding if available space
            if th - crop_h > 1:
                pad_h_before += np.random.randint(0, th - crop_h - pad_h_before + 1)
            if tw - crop_w > 1:
                pad_w_before += np.random.randint(0, tw - crop_w - pad_w_before + 1)
                
            # Ensure we don't go out of bounds
            pad_h_before = max(0, min(pad_h_before, th - crop_h))
            pad_w_before = max(0, min(pad_w_before, tw - crop_w))
            
            # Copy the data
            if len(data_array.shape) == 3:
                reshaped_data[pad_h_before:pad_h_before+crop_h, 
                             pad_w_before:pad_w_before+crop_w, :] = \
                    data_array[start_h:start_h+crop_h, start_w:start_w+crop_w, :]
            else:
                reshaped_data[pad_h_before:pad_h_before+crop_h, 
                             pad_w_before:pad_w_before+crop_w] = \
                    data_array[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        return reshaped_data

    def __iter__(self):
        date_buffer = {}  # Store date information for lead time pairing
        
        for data, variables, out_variables, metadata in self.dataset:
            date_str = metadata['date']
            region_id = metadata['region_id']
            
            # Process the current data
            input_data = []
            for var in variables:
                # Apply reshaping and append as a channel
                reshaped = self._reshape_to_target_size(data[var])
                input_data.append(reshaped)
                
            # Stack channels
            if len(input_data) > 0:
                input_data = np.stack(input_data, axis=0)  # [C, H, W]
                
                # Convert to tensor
                input_tensor = torch.from_numpy(input_data).float()
                
                # Store in buffer with date information
                key = (region_id, date_str)
                date_buffer[key] = {
                    'data': input_tensor,
                    'variables': variables,
                    'out_variables': variables,  # Use same variables for output
                    'date': datetime.strptime(date_str, "%Y-%m-%d")
                }
                
                # Find matching future dates for forecasting
                for stored_key, stored_data in list(date_buffer.items()):
                    stored_region, stored_date_str = stored_key
                    
                    # Only match within same region
                    if stored_region != region_id:
                        continue
                        
                    stored_date = stored_data['date']
                    current_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Calculate date difference
                    if current_date > stored_date:
                        days_diff = (current_date - stored_date).days
                        
                        # Check if within prediction range
                        if self.min_predict_range <= days_diff <= self.max_predict_range:
                            if not self.random_lead_time or (self.random_lead_time and random.random() < 0.5):
                                # Create a forecast pair
                                lead_time = torch.tensor(days_diff, dtype=torch.float32) / 10.0  # Scale to similar range
                                
                                yield (
                                    stored_data['data'],          # Input data
                                    input_tensor,                 # Target data
                                    lead_time,                    # Lead time in days
                                    stored_data['variables'],     # Input variables
                                    stored_data['variables']      # Output variables (same as input)
                                )
                
                # Clean up old entries
                # Remove entries that are too old to be used
                current_date = datetime.strptime(date_str, "%Y-%m-%d")
                for key in list(date_buffer.keys()):
                    stored_region, stored_date_str = key
                    stored_date = date_buffer[key]['date']
                    if (current_date - stored_date).days > self.max_predict_range:
                        del date_buffer[key]


class IndividualForecastDataIter(IterableDataset):
    """Apply transforms to forecast data."""
    def __init__(self, dataset, transforms, output_transforms):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for inp, out, lead_times, variables, out_variables in self.dataset:
            yield self.transforms(inp), self.output_transforms(out), lead_times, variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    """Shuffles data using a buffer."""
    def __init__(self, dataset, buffer_size):
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


def collate_fn(batch):
    """Collate function for the wildfire dataloader."""
    inputs = torch.stack([item[0] for item in batch])
    outputs = torch.stack([item[1] for item in batch])
    lead_times = torch.stack([item[2] for item in batch])
    
    # Variables should be the same for all items in the batch
    variables = batch[0][3]
    out_variables = batch[0][4]
    
    return inputs, outputs, lead_times, variables, out_variables


class WildfireDataModule(LightningDataModule):
    """Lightning DataModule for wildfire forecasting."""
    def __init__(
        self,
        root_dir,
        variables,
        out_variables=None,  # This parameter is kept for backward compatibility but not used
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        prediction_range=7,
        min_prediction_range=1,
        random_lead_time=False,
        batch_size=32,
        num_workers=4,
        buffer_size=1000,
        pin_memory=True,
        patch_size=(128, 128),
        valid_dates_csv=None,  # New parameter: path to CSV file with valid dates
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Calculate start and end indices
        self.train_start, self.train_end = 0.0, train_ratio
        self.val_start, self.val_end = train_ratio, train_ratio + val_ratio
        self.test_start, self.test_end = train_ratio + val_ratio, 1.0
        
        # Load valid dates from CSV if provided
        self.valid_dates = self._load_valid_dates_csv(valid_dates_csv) if valid_dates_csv else None
        
        # Discover files
        self.file_list = self._get_file_list(root_dir)
        print(f"Found {len(self.file_list)} NPZ files")
        
        # Create transforms
        self.transforms = self._get_normalize()
        self.output_transforms = self._get_normalize()  # Use same normalization for input and output
        
    def _load_valid_dates_csv(self, csv_path):
        """Load valid dates from CSV file."""
        print(f"Loading valid dates from CSV: {csv_path}")
        valid_date_patches = {}
        try:
            with open(csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        # Extract the date from the first column and patch ID from the second
                        date_str = row[0].strip()
                        patch_id = int(row[1].strip())
                        
                        # Store as a mapping from date to a set of valid patch IDs
                        if date_str not in valid_date_patches:
                            valid_date_patches[date_str] = set()
                        valid_date_patches[date_str].add(patch_id)
            
            print(f"Loaded valid date-patch pairs from CSV: {len(valid_date_patches)} dates")
            return valid_date_patches
        except Exception as e:
            print(f"Error loading valid dates CSV: {e}")
            return None
        
    def _get_file_list(self, root_dir):
        import glob
        npz_pattern = os.path.join(root_dir, "*.npz")
        file_list = glob.glob(npz_pattern)
        
        # Filter out known corrupt files
        corrupt_files = ["705_2023-11-27_2023-12-26.npz"]
        filtered_list = []
        for file_path in file_list:
            filename = os.path.basename(file_path)
            if filename not in corrupt_files:
                filtered_list.append(file_path)
            else:
                print(f"Skipping known corrupt file: {filename}")
        
        return filtered_list
        
    def _get_normalize(self, specific_variables=None):
        """Create normalization transform."""
        print(f"Getting normalization for variables: {specific_variables}")
        variables_to_use = self.hparams.variables  # Always use the input variables
        
        # Create a unique identifier for this set of variables
        vars_hash = hashlib.md5(str(variables_to_use).encode()).hexdigest()
        stats_filename = f"norm_stats_{vars_hash}.pkl"
        
        # Check if we have pre-computed statistics
        if os.path.exists(stats_filename):
            print(f"Loading normalization statistics from {stats_filename}")
            with open(stats_filename, 'rb') as f:
                stats = pickle.load(f)
                mean = stats['mean']
                std = stats['std']
        else:
            print("Computing normalization statistics from data samples...")
            # Sample data to compute normalization statistics
            sample_size = min(100, len(self.file_list))
            sample_files = np.random.choice(self.file_list, sample_size, replace=False)
            
            # Initialize arrays to store means and stds
            means = []
            stds = []
            
            # Process each variable
            for var_idx, var in enumerate(variables_to_use):
                all_values = []
                
                # Sample from files
                for file_path in sample_files:
                    try:
                        with np.load(file_path) as data:
                            if var in data:
                                var_data = data[var]
                                # Skip if all values are null/nan
                                if not np.all(np.isnan(var_data)):
                                    # Filter out nan values
                                    valid_data = var_data[~np.isnan(var_data)]
                                    if len(valid_data) > 0:
                                        all_values.append(valid_data)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
                
                # Compute statistics if we have data
                if all_values:
                    all_values = np.concatenate(all_values)
                    means.append(float(np.mean(all_values)))
                    stds.append(float(np.std(all_values) + 1e-6))  # Add small epsilon to avoid division by zero
                else:
                    # Default values if no valid data found
                    means.append(0.0)
                    stds.append(1.0)
            
            mean = torch.tensor(means)
            std = torch.tensor(stds)
            
            # Save statistics for future use
            with open(stats_filename, 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)
            print(f"Saved normalization statistics to {stats_filename}")
        
        return transforms.Normalize(mean=mean, std=std)
        
    def setup(self, stage=None):
        # Shuffle file list for reproducibility
        np.random.seed(42)
        np.random.shuffle(self.file_list)
        
    def train_dataloader(self):
        train_dataset = ShuffleIterableDataset(
            IndividualForecastDataIter(
                WildfireForecast(
                    WildfireNpzReader(
                        self.file_list,
                        start_idx=self.train_start,
                        end_idx=self.train_end,
                        variables=self.hparams.variables,
                        shuffle=True,
                        multi_dataset_training=True,
                        valid_dates=self.valid_dates,  # Pass valid dates to the reader
                    ),
                    max_predict_range=self.hparams.prediction_range,
                    min_predict_range=self.hparams.min_prediction_range,
                    random_lead_time=self.hparams.random_lead_time,
                ),
                self.transforms,
                self.output_transforms,
            ),
            self.hparams.buffer_size,
        )
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
        
    def val_dataloader(self):
        # If validation ratio is 0, return None
        if self.val_start >= self.val_end or abs(self.val_end - self.val_start) < 1e-5:
            print("INFO: Validation ratio is 0, returning None for validation dataloader")
            return None
            
        val_dataset = IndividualForecastDataIter(
            WildfireForecast(
                WildfireNpzReader(
                    self.file_list,
                    start_idx=self.val_start,
                    end_idx=self.val_end,
                    variables=self.hparams.variables,
                    shuffle=False,
                    valid_dates=self.valid_dates,  # Pass valid dates to the reader
                ),
                max_predict_range=self.hparams.prediction_range,
                min_prediction_range=self.hparams.min_prediction_range,
                random_lead_time=False,  # No random lead time for validation
            ),
            self.transforms,
            self.output_transforms,
        )
        
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
        
    def test_dataloader(self):
        # If test ratio is 0, return None
        if self.test_start >= self.test_end or abs(self.test_end - self.test_start) < 1e-5:
            print("INFO: Test ratio is 0, returning None for test dataloader")
            return None
            
        test_dataset = IndividualForecastDataIter(
            WildfireForecast(
                WildfireNpzReader(
                    self.file_list,
                    start_idx=self.test_start,
                    end_idx=self.test_end,
                    variables=self.hparams.variables,
                    shuffle=False,
                    valid_dates=self.valid_dates,  # Pass valid dates to the reader
                ),
                max_predict_range=self.hparams.prediction_range,
                min_prediction_range=self.hparams.min_prediction_range,
                random_lead_time=False,  # No random lead time for testing
            ),
            self.transforms,
            self.output_transforms,
        )
        
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
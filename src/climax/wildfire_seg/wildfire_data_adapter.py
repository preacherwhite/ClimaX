import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, RandomSampler

class WildfireDataAdapter:
    """Adapter to convert FireSpreadDataset outputs to ClimaX-compatible inputs.
    
    This adapter processes outputs from FireSpreadDataset and reformats them
    to be compatible with the ClimaX model architecture for segmentation tasks.
    
    Args:
        num_leading_observations (int): Number of leading observations/time steps
        flatten_temporal_dimension (bool): Whether to flatten temporal dimension
        img_size (tuple): Target image size for ClimaX
    """
    
    def __init__(
        self, 
        num_leading_observations: int = 1,
        flatten_temporal_dimension: bool = True,
        img_size: Tuple[int, int] = (128, 128)
    ):
        self.num_leading_observations = num_leading_observations
        self.flatten_temporal_dimension = flatten_temporal_dimension
        self.img_size = img_size
    
    def __call__(self, batch):
        """Transform a batch of data to match ClimaX's expected format."""
        # Unpack the batch into x and y
        x, y = zip(*batch)
        x = torch.stack(x)  # Shape: [B, T, C, H, W]
        y = torch.stack(y)  # Shape: [B, H, W]

        # Handle temporal dimension
        if self.flatten_temporal_dimension:
            # Reshape from [B, T, C, H, W] to [B, T*C, H, W]
            B, T, C, H, W = x.shape
            x = x.reshape(B, T*C, H, W)
        else:
            # Keep temporal dimension and use last time step
            x = x[:, -1, :, :, :]  # Shape: [B, C, H, W]

        # Check if the input dimensions match the expected size
        if tuple(x.shape[2:]) != tuple(self.img_size):
            raise ValueError(f"Input image size {x.shape[2:]} does not match expected size {self.img_size}")

        # Create lead times tensor (assuming we're predicting next day)
        lead_times = torch.tensor([1] * len(x), dtype=torch.float32)

        # Create variables list (matching the features in the data)
        variables = [
            "M11", "I2", "I1", "NDVI_last", "EVI2_last", "total precipitation",
            "wind speed", "wind direction", "minimum temperature", "maximum temperature",
            "energy release component", "specific humidity", "slope", "aspect",
            "elevation", "pdsi", "LC_Type1", "forecast total precipitation",
            "forecast wind speed", "forecast wind direction", "forecast temperature",
            "forecast specific humidity"
        ]

        # For segmentation, we only predict the active fire mask
        out_variables = ["active_fire"]  # The last channel of the last image is the active fire mask

        return x, y, lead_times, variables, out_variables


class ClimaXFireSpreadDataModule(LightningDataModule):
    """Wrapper for FireSpreadDataModule to make it compatible with ClimaX.
    
    This wrapper adapts the FireSpreadDataModule to be compatible with the 
    ClimaX model architecture by reformatting the data outputs.
    
    Args:
        fire_spread_datamodule: The original FireSpreadDataModule instance
        flatten_temporal_dimension (bool): Whether temporal dimension is flattened
        img_size (tuple): Target image size for ClimaX
        variable_mapping (dict): Mapping from indices to variable names
    """
    
    def __init__(
        self, 
        fire_spread_datamodule: LightningDataModule,
        flatten_temporal_dimension: bool = True,
        img_size: Tuple[int, int] = (128, 128),
        variable_mapping: Optional[Dict[int, str]] = None
    ):
        super().__init__()
        self.fire_datamodule = fire_spread_datamodule
        self.adapter = WildfireDataAdapter(
            num_leading_observations=fire_spread_datamodule.n_leading_observations,
            flatten_temporal_dimension=flatten_temporal_dimension,
            img_size=img_size
        )
        
        # Map feature indices to variable names for ClimaX
        if variable_mapping is None:
            # Use default mapping from FireSpreadDataset
            from dataloader.FireSpreadDataset import FireSpreadDataset
            self.variable_mapping = FireSpreadDataset.map_channel_index_to_features(only_base=False)
        else:
            self.variable_mapping = variable_mapping
        
        # Convert variable mapping to a list of strings
        self.variables = [self.variable_mapping[i] for i in range(len(self.variable_mapping))]
    
    def setup(self, stage=None):
        """Set up the dataset for training, validation, and testing."""
        self.fire_datamodule.setup(stage)
    
    def get_lat_lon(self):
        """Return latitude and longitude arrays."""
        # Use placeholder values or calculate from fire dataset
        # Simple dummy implementation:
        h, w = self.adapter.img_size
        # Approximate CONUS range for USA - adjust if needed
        lat = torch.linspace(24, 50, h)
        lon = torch.linspace(-125, -66, w)
        return lat, lon
    
    def get_climatology(self, partition='val'):
        """Get the climatology tensor for a partition."""
        # Placeholder implementation - could be enhanced with real data
        return torch.zeros(len(self.variables))
    
    def train_dataloader(self):
        """Get the training data loader with adapter wrapper."""
        loader = self.fire_datamodule.train_dataloader()
        return self._wrap_dataloader(loader)
    
    def val_dataloader(self):
        """Get the validation data loader with adapter wrapper."""
        loader = self.fire_datamodule.val_dataloader()
        return self._wrap_dataloader(loader)
    
    def test_dataloader(self):
        """Get the test data loader with adapter wrapper."""
        loader = self.fire_datamodule.test_dataloader()
        return self._wrap_dataloader(loader)
    
    def predict_dataloader(self):
        """Get the prediction data loader with adapter wrapper."""
        loader = self.fire_datamodule.predict_dataloader()
        return self._wrap_dataloader(loader)
    
    def _wrap_dataloader(self, loader):
        """Wrap the dataloader to handle the batch format."""
        if isinstance(loader, DataLoader):
            # Get the original shuffle setting from the dataset's sampler
            shuffle = isinstance(loader.sampler, RandomSampler)
        else:
            shuffle = False

        return DataLoader(
            dataset=loader.dataset,
            batch_size=loader.batch_size,
            num_workers=loader.num_workers,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )

    def _collate_fn(self, batch):
        """Custom collate function that applies the adapter to the batch."""
        # Apply the original collate function if available
        if hasattr(self.fire_datamodule, '_collate_fn') and self.fire_datamodule._collate_fn is not None:
            batch = self.fire_datamodule._collate_fn(batch)
        
        # Apply our adapter
        return self.adapter(batch)
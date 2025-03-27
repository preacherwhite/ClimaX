import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from pytorch_lightning import LightningModule
import warnings # For scheduler fallback warning

# Assuming these imports point to the correct locations in your project structure
from climax.arch import ClimaX
from climax.utils.metrics import mse, mse_val, rmse_val # Assumes these are defined correctly
from climax.utils.pos_embed import interpolate_pos_embed # Assumes this is defined correctly

class WildfireModule(LightningModule):
    """
    LightningModule for training ClimaX on wildfire prediction tasks.
    Uses an epoch-based learning rate scheduler with linear warmup and cosine decay.
    """

    def __init__(
        self,
        net: ClimaX,
        pretrained_path: str = "",
        lr: float = 1e-4, # Adjusted default based on previous config
        beta_1: float = 0.9,
        beta_2: float = 0.95,
        weight_decay: float = 1e-5,
        # --- Epoch-Based Scheduler Parameters ---
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        warmup_start_factor: float = 0.01, # Start LR = lr * factor
        eta_min: float = 1e-8,          # Min LR for cosine decay phase
        # --- End Scheduler Parameters ---
        prediction_range: int = 7, # Default based on previous config
    ):
        """
        Args:
            net (ClimaX): The underlying ClimaX model architecture.
            pretrained_path (str, optional): Path to pre-trained model weights. Defaults to "".
            lr (float, optional): Peak learning rate. Defaults to 1e-4.
            beta_1 (float, optional): AdamW beta1. Defaults to 0.9.
            beta_2 (float, optional): AdamW beta2. Defaults to 0.95.
            weight_decay (float, optional): AdamW weight decay. Defaults to 1e-5.
            warmup_epochs (int, optional): Number of epochs for linear warmup. Defaults to 5.
            max_epochs (int, optional): Total number of training epochs (must match trainer). Defaults to 100.
            warmup_start_factor (float, optional): Factor to multiply initial LR by at start of warmup. Defaults to 0.01.
            eta_min (float, optional): Minimum learning rate during cosine decay. Defaults to 1e-8.
            prediction_range (int, optional): Max prediction range (can be overridden by set_pred_range). Defaults to 7.
        """
        super().__init__()
        # Save hyperparameters, ignoring the potentially large 'net' object
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

        # Initialize placeholders for data module properties (set via methods)
        self.lat = None
        self.lon = None
        self.val_clim = None
        self.test_clim = None
        # Use prediction_range from hparams initially, can be updated by set_pred_range
        self.pred_range = self.hparams.prediction_range

        # Placeholders for potential denormalization (if needed for analysis/logging)
        self.denorm_mean = None
        self.denorm_std = None


    def load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights from a checkpoint file or URL."""
        print(f"INFO: Loading pre-trained checkpoint from: {pretrained_path}")
        if pretrained_path.startswith("http"):
           checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location="cpu")
        else:
           checkpoint = torch.load(pretrained_path, map_location="cpu")

        if "state_dict" not in checkpoint:
             print(f"WARNING: Checkpoint does not contain 'state_dict' key. Assuming entire checkpoint is the state dict.")
             checkpoint_model = checkpoint
        else:
             checkpoint_model = checkpoint["state_dict"]

        # Interpolate positional embeddings if needed
        try:
             interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)
        except Exception as e:
             print(f"WARNING: Could not interpolate positional embeddings: {e}")

        state_dict = self.state_dict()

        # Rename keys if needed (e.g., channel -> var) - adjust based on actual key names
        renamed_checkpoint_model = {}
        for k, v in checkpoint_model.items():
             new_k = k
             if "channel" in k:
                  new_k = k.replace("channel", "var")
                  print(f"INFO: Renaming checkpoint key {k} -> {new_k}")
             renamed_checkpoint_model[new_k] = v
        checkpoint_model = renamed_checkpoint_model

        # Filter out incompatible keys (missing or mismatched shape)
        filtered_checkpoint_model = {}
        for k, v in checkpoint_model.items():
            if k in state_dict and v.shape == state_dict[k].shape:
                filtered_checkpoint_model[k] = v
            else:
                print(f"INFO: Removing key '{k}' from pretrained checkpoint due to mismatch (present: {k in state_dict}, shape: {v.shape if k in state_dict else 'N/A'} vs expected: {state_dict.get(k, None)}).")


        # Load the filtered state dict
        msg = self.load_state_dict(filtered_checkpoint_model, strict=False)
        print(f"INFO: Weight loading message: {msg}")


    def set_denormalization(self, mean, std):
        """Store denormalization parameters (mean and std)."""
        # Convert to tensors and store on device later if needed, store as is for now
        self.denorm_mean = torch.tensor(mean) if mean is not None else None
        self.denorm_std = torch.tensor(std) if std is not None else None
        print("INFO: Denormalization parameters stored.")


    def set_lat_lon(self, lat, lon):
        """Store latitude and longitude arrays as tensors."""
        # Store as tensors, move to device within steps if needed
        self.lat = torch.tensor(lat, dtype=torch.float32) if lat is not None else None
        self.lon = torch.tensor(lon, dtype=torch.float32) if lon is not None else None
        print(f"INFO: Lat/lon stored (Lat shape: {self.lat.shape if self.lat is not None else 'None'}, Lon shape: {self.lon.shape if self.lon is not None else 'None'}).")


    def set_pred_range(self, r):
        """Set the prediction range (used for logging purposes)."""
        self.pred_range = r
        self.hparams.prediction_range = r # Also update hparams if needed elsewhere
        print(f"INFO: Prediction range set to: {self.pred_range}")


    def set_val_clim(self, clim):
        """Store validation climatology tensor."""
        # Store as tensor, move to device within steps if needed
        self.val_clim = clim.clone().detach() if clim is not None else None
        print(f"INFO: Validation climatology stored (shape: {self.val_clim.shape if self.val_clim is not None else 'None'}).")


    def set_test_clim(self, clim):
        """Store test climatology tensor."""
        # Store as tensor, move to device within steps if needed
        self.test_clim = clim.clone().detach() if clim is not None else None
        print(f"INFO: Test climatology stored (shape: {self.test_clim.shape if self.test_clim is not None else 'None'}).")


    def forward(self, x, y, lead_times, variables, out_variables):
        """Direct forward pass through the network."""
        # Wraps the core network's forward pass
        # Assuming the first element of the return tuple is the loss dict
        loss_dict, predictions = self.net.forward(x, y, lead_times, variables, out_variables, [mse], lat=self.lat.to(x.device) if self.lat is not None else None)
        return loss_dict[0], predictions


    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        x, y, lead_times, variables, out_variables = batch
        lat = self.lat.to(x.device) if self.lat is not None else None

        # Assuming net.forward returns [loss_dict], predictions
        loss_dict, _ = self.net.forward(x, y, lead_times, variables, out_variables, [mse], lat=lat)
        if isinstance(loss_dict, list): loss_dict = loss_dict[0] # Ensure it's a dict

        # Log training loss components
        for var, value in loss_dict.items():
            self.log(
                f"train/{var}",
                value,
                on_step=True,   # Log instantaneous step value
                on_epoch=True,  # Log average epoch value
                prog_bar=(var == "loss"), # Show main loss in progress bar
                sync_dist=True  # Aggregate correctly across devices
            )

        # Return the main loss for optimization
        if "loss" not in loss_dict:
             warnings.warn("Key 'loss' not found in loss_dict returned by net.forward. Using first value.")
             return next(iter(loss_dict.values())) # Return first value as loss
        return loss_dict["loss"]


    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        x, y, lead_times, variables, out_variables = batch
        # Use mean lead time or hparam for consistent logging tag? Using hparam.
        log_postfix = f"_day_{int(self.hparams.prediction_range)}"

        metrics_to_compute = [mse_val, rmse_val] # Define metrics for evaluation

        # Ensure auxiliary data is on the correct device
        lat = self.lat.to(x.device) if self.lat is not None else None
        val_clim = self.val_clim.to(x.device) if self.val_clim is not None else None

        # Assuming net.evaluate returns a list of dictionaries (one per metric)
        all_loss_dicts = self.net.evaluate(
            x, y, lead_times, variables, out_variables,
            transform=None, # Assuming data is pre-normalized
            metrics=metrics_to_compute,
            lat=lat,
            clim=val_clim,
            log_postfix=log_postfix
        )

        # Combine results from different metrics into one dictionary
        combined_loss_dict = {}
        for d in all_loss_dicts:
            combined_loss_dict.update(d)

        # Log validation metrics (epoch level)
        for var, value in combined_loss_dict.items():
            self.log(
                f"val/{var}", # Metric names should come from evaluate()
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(f"rmse{log_postfix}" in var), # Show RMSE in progress bar
                sync_dist=True # Important for DDP validation
            )

        # Return the dictionary for potential use in callbacks (e.g., ModelCheckpoint)
        return combined_loss_dict


    def test_step(self, batch, batch_idx):
        """Performs a single test step."""
        x, y, lead_times, variables, out_variables = batch
        log_postfix = f"_day_{int(self.hparams.prediction_range)}"
        metrics_to_compute = [mse_val, rmse_val]

        # Ensure auxiliary data is on the correct device
        lat = self.lat.to(x.device) if self.lat is not None else None
        test_clim = self.test_clim.to(x.device) if self.test_clim is not None else None

        all_loss_dicts = self.net.evaluate(
            x, y, lead_times, variables, out_variables,
            transform=None,
            metrics=metrics_to_compute,
            lat=lat,
            clim=test_clim,
            log_postfix=log_postfix
        )

        combined_loss_dict = {}
        for d in all_loss_dicts:
           combined_loss_dict.update(d)

        # Log test metrics (epoch level)
        for var, value in combined_loss_dict.items():
            self.log(
                f"test/{var}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True # Important for DDP test aggregation
            )
        return combined_loss_dict


    def configure_optimizers(self):
        """Configure optimizer (AdamW) and epoch-based LR scheduler (SequentialLR)."""
        # --- Optimizer setup (AdamW with parameter groups) ---
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue # Skip parameters that don't require gradients

            # Separate params for weight decay handling
            # Common practice: No decay for biases, normalization layers, embeddings
            if param.ndim <= 1 or "bias" in name or "norm" in name or "embed" in name:
                # print(f"No weight decay for: {name}")
                no_decay.append(param)
            else:
                # print(f"Weight decay for: {name}")
                decay.append(param)

        if not decay and not no_decay:
             raise ValueError("No parameters requiring gradients found.")

        # Create parameter groups for optimizer
        param_groups = []
        if decay:
             param_groups.append({
                 "params": decay,
                 "lr": self.hparams.lr, # Use peak LR here
                 "betas": (self.hparams.beta_1, self.hparams.beta_2),
                 "weight_decay": self.hparams.weight_decay,
             })
        if no_decay:
             param_groups.append({
                 "params": no_decay,
                 "lr": self.hparams.lr, # Use peak LR here
                 "betas": (self.hparams.beta_1, self.hparams.beta_2),
                 "weight_decay": 0.0, # Explicitly set no weight decay
             })

        optimizer = torch.optim.AdamW(param_groups)
        # --- End Optimizer setup ---


        # --- Configure Epoch-Based Scheduler (Warmup + Cosine Decay) ---
        print(f"INFO: Setting up epoch-based scheduler: Linear Warmup ({self.hparams.warmup_epochs} epochs, start_factor={self.hparams.warmup_start_factor}) -> Cosine Decay ({self.hparams.max_epochs - self.hparams.warmup_epochs} epochs, eta_min={self.hparams.eta_min})")

        # 1. Warmup Scheduler (LinearLR)
        # Runs for warmup_epochs epochs, scaling LR from start_factor*lr to 1.0*lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_start_factor,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs # number of *epochs* for warmup
        )

        # 2. Cosine Decay Scheduler (CosineAnnealingLR)
        # Calculate the number of epochs remaining for cosine decay
        cosine_epochs = max(1, self.hparams.max_epochs - self.hparams.warmup_epochs) # T_max must be > 0

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs, # number of *epochs* for decay
            eta_min=self.hparams.eta_min # minimum learning rate
        )

        # 3. Combine with SequentialLR
        # The milestone is the epoch *after which* the second scheduler starts.
        # If warmup_epochs is 5, milestone is 5 -> Cosine starts at epoch 6.
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs]
        )
        # --- End Scheduler Configuration ---

        # Configure the dictionary for PyTorch Lightning
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch", # Tell Lightning to step the scheduler per epoch
            "frequency": 1,      # Step frequency is 1 (per interval=epoch)
            # "monitor": "val/loss", # Optional: Monitor a metric for ReduceLROnPlateau, not needed here
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
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
        # --- Step-Based Scheduler Parameters ---
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        warmup_start_factor: float = 0.01, # Start LR = lr * factor
        eta_min: float = 1e-8,          # Min LR for cosine decay phase
        # --- End Scheduler Parameters ---
        prediction_range: int = 7, # Default based on previous config
        excluded_vars: list = None, # Add excluded variables parameter
    ):
        """
        Args:
            net (ClimaX): The underlying ClimaX model architecture.
            pretrained_path (str, optional): Path to pre-trained model weights. Defaults to "".
            lr (float, optional): Peak learning rate. Defaults to 1e-4.
            beta_1 (float, optional): AdamW beta1. Defaults to 0.9.
            beta_2 (float, optional): AdamW beta2. Defaults to 0.95.
            weight_decay (float, optional): AdamW weight decay. Defaults to 1e-5.
            warmup_steps (int, optional): Number of steps for linear warmup. Defaults to 1000.
            max_steps (int, optional): Total number of training steps (must match trainer). Defaults to 100000.
            warmup_start_factor (float, optional): Factor to multiply initial LR by at start of warmup. Defaults to 0.01.
            eta_min (float, optional): Minimum learning rate during cosine decay. Defaults to 1e-8.
            prediction_range (int, optional): Max prediction range (can be overridden by set_pred_range). Defaults to 7.
            excluded_vars (list, optional): List of variable names to be excluded during training/inference. Defaults to None.
        """
        super().__init__()
        # Save hyperparameters, ignoring the potentially large 'net' object
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        
        # Initialize excluded_vars if not provided
        self.excluded_vars = excluded_vars if excluded_vars is not None else []
        print(f"INFO: Excluded variables: {self.excluded_vars}")

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
        
        
    def set_excluded_vars(self, excluded_vars):
        """Set the list of variables to exclude from processing."""
        self.excluded_vars = excluded_vars if excluded_vars is not None else []
        print(f"INFO: Updated excluded variables: {self.excluded_vars}")
        
    def prepare_excluded_variables(self):
        """
        Prepare model for variable exclusion before training starts.
        
        This method does NOT modify the model architecture directly, as that would 
        require deep knowledge of the specific ClimaX implementation and potentially
        break compatibility. Instead, it prepares tracking information for proper
        runtime filtering.
        """
        if not self.excluded_vars or len(self.excluded_vars) == 0:
            print("INFO: No variables to exclude, model will use all variables.")
            return
            
        print(f"INFO: Preparing to exclude variables during processing: {self.excluded_vars}")
        
        # Instead of modifying architecture, we'll set up tracking for runtime filtering
        self._should_filter = True
        
        # Generate variable map for quick lookups during runtime
        if hasattr(self.net, 'default_vars'):
            self._var_indices = {var: idx for idx, var in enumerate(self.net.default_vars)}
            self._excluded_indices = [self._var_indices[var] for var in self.excluded_vars 
                                     if var in self._var_indices]
            
            if len(self._excluded_indices) != len(self.excluded_vars):
                print(f"WARNING: Some excluded variables not found in model's variables. "
                      f"Found {len(self._excluded_indices)} of {len(self.excluded_vars)}.")
                
            print(f"INFO: Prepared to exclude indices: {self._excluded_indices}")
            
            # Store original variable count for verification
            self._original_var_count = len(self.net.default_vars)
        else:
            print("WARNING: Model does not have default_vars attribute. Variable exclusion will rely on runtime names.")
            self._var_indices = {}
            self._excluded_indices = []
            self._original_var_count = 0
        
        print("INFO: Variable exclusion will be applied during runtime processing")


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


    def _filter_excluded_variables(self, x, y, variables, out_variables):
        """
        Remove excluded variable channels from both input and target tensors.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W)
            y (torch.Tensor): Target tensor with shape (B, C, H, W)
            variables (list): List of variable names corresponding to channels
            out_variables (list): List of output variable names
            
        Returns:
            tuple: (filtered_x, filtered_y, filtered_variables, filtered_out_variables)
        """
        if not self.excluded_vars or len(self.excluded_vars) == 0:
            return x, y, variables, out_variables
            
        # Create masks for variables to keep
        keep_indices = []
        filtered_variables = []
        
        for i, var in enumerate(variables):
            if var not in self.excluded_vars:
                keep_indices.append(i)
                filtered_variables.append(var)
        
        # If all variables would be excluded, keep original
        if len(keep_indices) == 0:
            print("WARNING: All variables would be excluded. Keeping original tensors.")
            return x, y, variables, out_variables
            
        # Select only the non-excluded channels
        filtered_x = x[:, keep_indices]
        filtered_y = y[:, keep_indices]
        
        # Also filter output variables to match
        filtered_out_variables = [var for var in out_variables if var not in self.excluded_vars]
        if len(filtered_out_variables) == 0 and out_variables:
            # If all output variables would be excluded, keep all that remain in filtered_variables
            filtered_out_variables = filtered_variables
        
        # Only log this message once to avoid console spam
        if not hasattr(self, '_filter_logged'):
            excluded_count = len(variables) - len(filtered_variables)
            print(f"INFO: Removed {excluded_count} excluded variable channels. Kept {len(filtered_variables)} variables.")
            self._filter_logged = True
            
        return filtered_x, filtered_y, filtered_variables, filtered_out_variables


    def forward(self, x, y, lead_times, variables, out_variables):
        """Direct forward pass through the network with variable filtering."""
        # Remove excluded variable channels from both input and target
        x_filtered, y_filtered, variables_filtered, out_variables_filtered = self._filter_excluded_variables(
            x, y, variables, out_variables
        )

        # Forward pass with filtered inputs, targets, and variable lists
        loss_dict, predictions = self.net.forward(
            x_filtered, y_filtered, lead_times, variables_filtered, out_variables_filtered, 
            [mse], lat=self.lat.to(x.device) if self.lat is not None else None
        )
        
        # Handle case where loss_dict might be a list
        if isinstance(loss_dict, list):
            loss_dict = loss_dict[0]
            
        return loss_dict, predictions


    def training_step(self, batch, batch_idx):
        """Performs a single training step with variable filtering."""
        x, y, lead_times, variables, out_variables = batch
        
        # Remove excluded variable channels from both input and target
        x_filtered, y_filtered, variables_filtered, out_variables_filtered = self._filter_excluded_variables(
            x, y, variables, out_variables
        )
        
        lat = self.lat.to(x.device) if self.lat is not None else None

        # Forward pass with filtered tensors and variable lists
        loss_dict, _ = self.net.forward(
            x_filtered, y_filtered, lead_times, variables_filtered, out_variables_filtered, 
            [mse], lat=lat
        )
        
        if isinstance(loss_dict, list): 
            loss_dict = loss_dict[0] # Ensure it's a dict

        # Log training loss components
        for var, value in loss_dict.items():
            self.log(
                f"train/{var}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(var == "loss"),
                sync_dist=True
            )

        # Return the main loss for optimization
        if "loss" not in loss_dict:
             warnings.warn("Key 'loss' not found in loss_dict returned by net.forward. Using first value.")
             return next(iter(loss_dict.values())) # Return first value as loss
        return loss_dict["loss"]


    def validation_step(self, batch, batch_idx):
        """Performs a single validation step with variable filtering."""
        # Skip validation when no validation data is available
        if batch is None:
            return None
            
        x, y, lead_times, variables, out_variables = batch
        # Remove excluded variable channels from both input and target
        x_filtered, y_filtered, variables_filtered, out_variables_filtered = self._filter_excluded_variables(
            x, y, variables, out_variables
        )
        # Use mean lead time or hparam for consistent logging tag
        log_postfix = f"_day_{int(self.hparams.prediction_range)}"

        metrics_to_compute = [mse_val, rmse_val] # Define metrics for evaluation

        # Ensure auxiliary data is on the correct device
        lat = self.lat.to(x.device) if self.lat is not None else None
        val_clim = self.val_clim.to(x.device) if self.val_clim is not None else None
        
        # If we have a climatology tensor, filter it to match our filtered variables
        if val_clim is not None and len(val_clim) == len(variables):
            keep_indices = [i for i, var in enumerate(variables) if var not in self.excluded_vars]
            if keep_indices:
                val_clim = val_clim[keep_indices]

        # Forward pass with filtered tensors and variable lists
        all_loss_dicts = self.net.evaluate(
            x_filtered, y_filtered, lead_times, variables_filtered, out_variables_filtered,
            transform=None,
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
                f"val/{var}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(f"rmse{log_postfix}" in var),
                sync_dist=True
            )

        # Return the dictionary for potential use in callbacks
        return combined_loss_dict


    def test_step(self, batch, batch_idx):
        """Performs a single test step with variable filtering."""
        # Skip testing when no test data is available
        if batch is None:
            return None
            
        x, y, lead_times, variables, out_variables = batch
        
        # Remove excluded variable channels from both input and target
        x_filtered, y_filtered, variables_filtered, out_variables_filtered = self._filter_excluded_variables(
            x, y, variables, out_variables
        )
        
        log_postfix = f"_day_{int(self.hparams.prediction_range)}"
        metrics_to_compute = [mse_val, rmse_val]

        # Ensure auxiliary data is on the correct device
        lat = self.lat.to(x.device) if self.lat is not None else None
        test_clim = self.test_clim.to(x.device) if self.test_clim is not None else None
        
        # If we have a climatology tensor, filter it to match our filtered variables
        if test_clim is not None and len(test_clim) == len(variables):
            keep_indices = [i for i, var in enumerate(variables) if var not in self.excluded_vars]
            if keep_indices:
                test_clim = test_clim[keep_indices]

        # Forward pass with filtered tensors and variable lists
        all_loss_dicts = self.net.evaluate(
            x_filtered, y_filtered, lead_times, variables_filtered, out_variables_filtered,
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
                sync_dist=True
            )
                
        return combined_loss_dict


    def configure_optimizers(self):
        """Configure optimizer (AdamW) and step-based LR scheduler (SequentialLR)."""
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


        # --- Configure Step-Based Scheduler (Warmup + Cosine Decay) ---
        print(f"INFO: Setting up step-based scheduler: Linear Warmup ({self.hparams.warmup_steps} steps, start_factor={self.hparams.warmup_start_factor}) -> "
              f"Cosine Decay ({self.hparams.max_steps - self.hparams.warmup_steps} steps, eta_min={self.hparams.eta_min})")

        # 1. Warmup Scheduler (LinearLR)
        # Runs for warmup_steps steps, scaling LR from start_factor*lr to 1.0*lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_start_factor,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps # number of *steps* for warmup
        )

        # 2. Cosine Decay Scheduler (CosineAnnealingLR)
        # Calculate the number of steps remaining for cosine decay
        cosine_steps = max(1, self.hparams.max_steps - self.hparams.warmup_steps) # T_max must be > 0

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps, # number of *steps* for decay
            eta_min=self.hparams.eta_min # minimum learning rate
        )

        # 3. Combine with SequentialLR
        # The milestone is the step *after which* the second scheduler starts.
        # If warmup_steps is 1000, milestone is 1000 -> Cosine starts at step 1001.
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps]
        )
        # --- End Scheduler Configuration ---

        # Configure the dictionary for PyTorch Lightning
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step", # Tell Lightning to step the scheduler per step (not per epoch)
            "frequency": 1,     # Step frequency is 1 (per interval=step)
            # "monitor": "val/loss", # Optional: Monitor a metric for ReduceLROnPlateau, not needed here
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
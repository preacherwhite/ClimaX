import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from climax.arch import ClimaX
from climax.utils.pos_embed import interpolate_pos_embed
from climax.utils.metrics import mse, lat_weighted_mse

class ClimaXWildfireSegmentation(ClimaX):
    """ClimaX model adapted for wildfire segmentation tasks.
    
    This model modifies the ClimaX architecture to output segmentation masks
    for wildfire prediction. The key modifications include:
    1. Using a decoder-based segmentation head instead of the standard prediction head
    2. Supporting binary or multi-class segmentation outputs
    3. Handling the specific feature structure of wildfire datasets
    
    Args:
        default_vars (list): List of default variables used for training
        img_size (list): Image size of the input data
        patch_size (int): Patch size of the input data
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer layers
        decoder_depth (int): Number of decoder layers
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension
        drop_path (float): Stochastic depth rate
        drop_rate (float): Dropout rate
        num_classes (int): Number of segmentation classes (default: 1 for binary segmentation)
        parallel_patch_embed (bool): Whether to use parallel patch embedding
    """
    def __init__(
        self,
        default_vars = [
          "M11", "I2", "I1", "NDVI_last", "EVI2_last", "total precipitation",
          "wind speed", "wind direction", "minimum temperature", "maximum temperature",
          "energy release component", "specific humidity", "slope", "aspect",
          "elevation", "pdsi", "LC_Type1", "forecast total precipitation",
          "forecast wind speed", "forecast wind direction", "forecast temperature",
          "forecast specific humidity"
        ],
        img_size=[128, 128],
        patch_size=16,
        embed_dim=768,
        depth=12,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        num_classes=1,
        parallel_patch_embed=False,
    ):
        super().__init__(
            default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            parallel_patch_embed=parallel_patch_embed
        )
        
        # Replace the standard prediction head with a segmentation head
        # Remove the existing head
        del self.head
        
        # Create a segmentation-specific decoder head
        segmentation_layers = []
        for _ in range(decoder_depth):
            segmentation_layers.append(nn.Linear(embed_dim, embed_dim))
            segmentation_layers.append(nn.GELU())
        
        # Final layer outputs num_classes * patch_size^2 values per patch
        segmentation_layers.append(nn.Linear(embed_dim, num_classes * patch_size**2))
        self.segmentation_head = nn.Sequential(*segmentation_layers)

    def forward_segmentation(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        """Forward pass for segmentation.
        
        Args:
            x: Input tensor of shape [B, V, H, W]
            lead_times: Lead time tensor of shape [B]
            variables: List of variable names
            
        Returns:
            Segmentation mask of shape [B, num_classes, H, W]
        """
        # Encode the input using the standard ClimaX encoder
        features = self.forward_encoder(x, lead_times, variables)
        
        # Apply the segmentation head
        segmentation_logits = self.segmentation_head(features)
        
        # Reshape to [B, num_classes, H, W]
        batch_size = x.shape[0]
        segmentation_masks = self.unpatchify_segmentation(segmentation_logits)
        
        return segmentation_masks
    
    def unpatchify_segmentation(self, x: torch.Tensor):
        """Convert patched representation back to image space for segmentation.
        
        Args:
            x: Tensor of shape [B, L, num_classes * patch_size^2]
            
        Returns:
            Tensor of shape [B, num_classes, H, W]
        """
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        num_classes = x.shape[-1] // (p * p)
        
        # Reshape to [B, h, w, p, p, num_classes]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, num_classes))
        
        # Permute to [B, num_classes, h, p, w, p]
        x = torch.einsum("nhwpqc->nchpwq", x)
        
        # Reshape to [B, num_classes, H, W]
        segmentation_masks = x.reshape(shape=(x.shape[0], num_classes, h * p, w * p))
        
        return segmentation_masks

    def forward(self, x, y, lead_times, variables, metric, lat=None, class_weights=None):
        """Forward pass through the model for training.
        
        Args:
            x: Input tensor [B, V, H, W]
            y: Target segmentation mask [B, H, W]
            lead_times: Lead times tensor [B]
            variables: List of variable names
            metric: List of metric functions to compute
            lat: Optional latitude tensor for weighted metrics
            class_weights: Optional class weights for loss calculation
            
        Returns:
            tuple: (loss_dict, segmentation_masks)
        """
        # Get segmentation masks
        segmentation_masks = self.forward_segmentation(x, lead_times, variables)
        
        # For binary segmentation, apply sigmoid and compute BCE loss
        if segmentation_masks.shape[1] == 1:
            # Ensure y has the right shape for binary segmentation
            if len(y.shape) == 3:
                y = y.unsqueeze(1)
                
            # Compute loss and metrics
            loss_dict = {}
            
            # Binary segmentation loss (BCE)
            if class_weights is not None:
                pos_weight = torch.tensor([class_weights], device=x.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
                
            loss = criterion(segmentation_masks, y.float())
            loss_dict["loss"] = loss
            
            # Compute additional metrics if provided
            if metric is not None:
                for m in metric:
                    metric_dict = m(torch.sigmoid(segmentation_masks), y, ["segmentation"], lat)
                    loss_dict.update(metric_dict)
                    
        # For multi-class segmentation
        else:
            # Cross-entropy loss for multi-class
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(segmentation_masks, y.long())
            loss_dict = {"loss": loss}
            
            # Compute additional metrics if provided
            if metric is not None:
                pred_classes = torch.argmax(segmentation_masks, dim=1, keepdim=True)
                for m in metric:
                    metric_dict = m(pred_classes, y.unsqueeze(1), ["segmentation"], lat)
                    loss_dict.update(metric_dict)
        
        return loss_dict, segmentation_masks


class ClimaXWildfireSegmentationModule(LightningModule):
    """Lightning module for wildfire segmentation using ClimaX.
    
    Args:
        net (ClimaXWildfireSegmentation): The ClimaX segmentation model
        pretrained_path (str): Path to pretrained ClimaX checkpoint
        lr (float): Learning rate
        beta_1 (float): Beta 1 for AdamW optimizer
        beta_2 (float): Beta 2 for AdamW optimizer
        weight_decay (float): Weight decay for optimizer
        pos_class_weight (float): Positive class weight for binary segmentation
        num_classes (int): Number of segmentation classes
        warmup_epochs (int): Number of warmup epochs
        max_epochs (int): Maximum number of training epochs
    """
    
    def __init__(
        self,
        net: ClimaXWildfireSegmentation,
        pretrained_path: str = "",
        lr: float = 1e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.95,
        weight_decay: float = 1e-5,
        pos_class_weight: float = 1.0,
        num_classes: int = 1,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.num_classes = num_classes
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
            
        # Define class weights for loss function
        if num_classes == 1:  # Binary segmentation
            self.class_weights = pos_class_weight
        else:
            # For multi-class, initialize with equal weights
            # These can be adjusted based on class distribution
            self.class_weights = torch.ones(num_classes)
            
        # Metrics for tracking
        self.val_iou = 0.0
        self.val_f1 = 0.0
        
    def load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights from a checkpoint file or URL."""
        print(f"INFO: Loading pre-trained checkpoint from: {pretrained_path}")
        if pretrained_path.startswith("http"):
           checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location="cpu")
        else:
           checkpoint = torch.load(pretrained_path, map_location="cpu")

        if pretrained_path is None:
            print(f"INFO: No pre-trained weights provided.")
            return
        if "state_dict" not in checkpoint:
             print(f"WARNING: Checkpoint does not contain 'state_dict' key. Assuming entire checkpoint is the state dict.")
             checkpoint_model = checkpoint
        else:
             checkpoint_model = checkpoint["state_dict"]
        # Load the filtered state dict
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(f"INFO: Weight loading message: {msg}")
    
    def forward(self, x, lead_times, variables):
        """Forward pass for inference."""
        return self.net.forward_segmentation(x, lead_times, variables)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y, lead_times, variables, _ = batch
        
        loss_dict, segmentation_masks = self.net(
            x, 
            y, 
            lead_times, 
            variables, 
            metric=None, 
            class_weights=self.class_weights
        )
        
        # Log metrics
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate additional metrics for monitoring
        if self.num_classes == 1:  # Binary segmentation
            pred_binary = (torch.sigmoid(segmentation_masks) > 0.5).float()
            metrics = self._calculate_binary_metrics(pred_binary, y.unsqueeze(1) if len(y.shape) == 3 else y)
            
            for name, value in metrics.items():
                self.log(f"train/{name}", value, on_step=False, on_epoch=True)
        
        return loss_dict["loss"]
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y, lead_times, variables, _ = batch
        
        loss_dict, segmentation_masks = self.net(
            x, 
            y, 
            lead_times, 
            variables, 
            metric=None, 
            class_weights=self.class_weights
        )
        
        # Log metrics
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate additional metrics
        if self.num_classes == 1:  # Binary segmentation
            pred_binary = (torch.sigmoid(segmentation_masks) > 0.5).float()
            metrics = self._calculate_binary_metrics(pred_binary, y.unsqueeze(1) if len(y.shape) == 3 else y)
            
            for name, value in metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)
                
            # Store metrics for best model selection
            self.val_iou = metrics["iou"]
            self.val_f1 = metrics["f1"]
            
        return loss_dict["loss"]
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y, lead_times, variables, _ = batch
        
        loss_dict, segmentation_masks = self.net(
            x, 
            y, 
            lead_times, 
            variables, 
            metric=None, 
            class_weights=self.class_weights
        )
        
        # Log metrics
        self.log("test/loss", loss_dict["loss"], on_step=False, on_epoch=True)
        
        # Calculate additional metrics
        if self.num_classes == 1:  # Binary segmentation
            pred_binary = (torch.sigmoid(segmentation_masks) > 0.5).float()
            metrics = self._calculate_binary_metrics(pred_binary, y.unsqueeze(1) if len(y.shape) == 3 else y)
            
            for name, value in metrics.items():
                self.log(f"test/{name}", value, on_step=False, on_epoch=True)
                
        return {"loss": loss_dict["loss"], "preds": segmentation_masks, "targets": y}
    
    def _calculate_binary_metrics(self, pred, target):
        """Calculate metrics for binary segmentation."""
        # Ensure inputs have the same shape
        if pred.shape != target.shape:
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
        
        # Intersection over Union
        intersection = torch.sum(pred * target, dim=[1, 2, 3])
        union = torch.sum(pred, dim=[1, 2, 3]) + torch.sum(target, dim=[1, 2, 3]) - intersection
        iou = (intersection / (union + 1e-7)).mean()
        
        # Precision, Recall, F1
        tp = torch.sum(pred * target, dim=[1, 2, 3])
        fp = torch.sum(pred * (1 - target), dim=[1, 2, 3])
        fn = torch.sum((1 - pred) * target, dim=[1, 2, 3])
        
        precision = (tp / (tp + fp + 1e-7)).mean()
        recall = (tp / (tp + fn + 1e-7)).mean()
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for regularization
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name or "norm" in name or "bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0.0,
                },
            ]
        )
        
        # Linear warmup with cosine annealing
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.hparams.max_epochs,
                pct_start=self.hparams.warmup_epochs / self.hparams.max_epochs,
                anneal_strategy='cos',
                div_factor=25,
                final_div_factor=1e4,
            ),
            "interval": "epoch",
            "frequency": 1,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
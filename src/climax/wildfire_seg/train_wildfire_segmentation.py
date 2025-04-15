#!/usr/bin/env python
# Training script for ClimaX-based Wildfire Segmentation

import os
import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter

# Import the FireSpreadDataModule
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

# Import our custom ClimaX components
from climax_wildfire_segmentation import ClimaXWildfireSegmentation, ClimaXWildfireSegmentationModule
from wildfire_data_adapter import ClimaXFireSpreadDataModule

# Ensure proper environment variables
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


class ClimaXWildfireSegmentationCLI(LightningCLI):
    """Custom Lightning CLI for running wildfire segmentation with ClimaX."""
    
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool, default=True,
                            help="If True: train the model.")
        parser.add_argument("--do_predict", type=bool, default=False,
                            help="If True: compute predictions.")
        parser.add_argument("--do_test", type=bool, default=True,
                            help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool, 
                            default=True, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to checkpoint to load for resuming training, for testing and predicting.")
        parser.add_argument("--pretrained_weights", type=str, default=None,
                            help="Path to pretrained ClimaX weights for initialization.")
        parser.add_argument("--name", type=str, default="",
                            help="Name of the run.")
        
        # ClimaX specific arguments
        parser.add_argument("--img_size", nargs=2, type=int, default=[128, 128],
                            help="Image size for ClimaX model")
        parser.add_argument("--patch_size", type=int, default=8,
                            help="Patch size for ClimaX model")
        parser.add_argument("--embed_dim", type=int, default=768,
                            help="Embedding dimension for ClimaX model")
        parser.add_argument("--depth", type=int, default=8,
                            help="Transformer depth for ClimaX model")
        parser.add_argument("--decoder_depth", type=int, default=2,
                            help="Decoder depth for ClimaX model")
        parser.add_argument("--num_heads", type=int, default=8,
                            help="Number of attention heads for ClimaX model")
        parser.add_argument("--num_classes", type=int, default=1,
                            help="Number of segmentation classes (1 for binary)")
        parser.add_argument("--flatten_temporal_dimension", type=bool, default=True,
                            help="Whether to flatten the temporal dimension in the adapter")
    
    def before_instantiate_classes(self):
        """Prepare for model and datamodule instantiation."""
        # The number of features is determined by the dataset configuration
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.init_args.n_leading_observations,
            self.config.data.init_args.features_to_keep,
            self.config.data.init_args.remove_duplicate_features,
        )
        
        # Calculate positive class weight for wildfire segmentation
        # Non-fire pixels are marked as missing values in the active fire feature
        train_years, _, _ = FireSpreadDataModule.split_fires(
            self.config.data.init_args.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)
        
        # Generate default variable names based on FireSpreadDataset's mapping
        default_vars =[
          "M11", "I2", "I1", "NDVI_last", "EVI2_last", "total precipitation",
          "wind speed", "wind direction", "minimum temperature", "maximum temperature",
          "energy release component", "specific humidity", "slope", "aspect",
          "elevation", "pdsi", "LC_Type1", "forecast total precipitation",
          "forecast wind speed", "forecast wind direction", "forecast temperature",
          "forecast specific humidity"
        ]
        
        # Set up the ClimaX model configuration
        self.climax_config = {
            "default_vars": default_vars,
            "img_size": self.config.img_size,
            "patch_size": self.config.patch_size,
            "embed_dim": self.config.embed_dim,
            "depth": self.config.depth,
            "decoder_depth": self.config.decoder_depth,
            "num_heads": self.config.num_heads,
            "num_classes": self.config.num_classes,
            "mlp_ratio": 4.0,
            "drop_path": 0.1,
            "drop_rate": 0.1,
            "parallel_patch_embed": False,
        }
        
        # Set up the ClimaX module configuration
        self.climax_module_config = {
            "lr": self.config.model.init_args.lr if hasattr(self.config.model, 'init_args') and hasattr(self.config.model.init_args, 'lr') else 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.95,
            "weight_decay": 1e-5,
            "pos_class_weight": pos_class_weight,
            "num_classes": self.config.num_classes,
            "warmup_epochs": 5,
            "max_epochs": self.config.trainer.max_epochs,
            "pretrained_path": self.config.pretrained_weights if self.config.pretrained_weights else "",
        }

    @rank_zero_only
    def tensorboard_setup(self):
        """Set up tensorboard logging configuration."""
        tensorboard_dir = os.path.join(self.config.trainer.default_root_dir, "tensorboard")
        self.writer = SummaryWriter(tensorboard_dir)
        
        logger = self.trainer.logger
        if hasattr(logger, 'experiment'):
            # Add name suffix if provided
            experiment_name = logger.experiment.name
            if self.config.name:
                experiment_name += '_' + self.config.name
                
            # Add SLURM job ID if available
            if 'SLURM_JOB_ID' in os.environ:
                experiment_name += f'_{os.environ["SLURM_JOB_ID"]}'
            
            logger.experiment.name = experiment_name
        
        # Save configuration to file
        config_file_name = os.path.join(self.config.trainer.default_root_dir, "cli_config.yaml")
        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)

    def before_fit(self):
        """Set up before model fitting."""
        self.tensorboard_setup()
    
    def before_validate(self):
        """Set up before validation."""
        self.tensorboard_setup()
    
    def before_test(self):
        """Set up before testing."""
        self.tensorboard_setup()
        
    def setup_model_datamodule(self):
        """Set up the model and datamodule for training."""
        # Instantiate the original FireSpreadDataModule using the provided config
        fire_datamodule = FireSpreadDataModule(**self.config.data.init_args)
        
        # Wrap it with our adapter to make it compatible with ClimaX
        adapted_datamodule = ClimaXFireSpreadDataModule(
            fire_datamodule,
            flatten_temporal_dimension=self.config.flatten_temporal_dimension,
            img_size=self.config.model.init_args.net.init_args.img_size
        )
        
        # Instantiate the ClimaX segmentation model
        climax_model = ClimaXWildfireSegmentation(**self.climax_config)
        
        # Wrap it with the LightningModule for training
        climax_module = ClimaXWildfireSegmentationModule(
            net=climax_model,
            **self.climax_module_config
        )
        
        return climax_module, adapted_datamodule
        
        
def main():
    """Main entry point for training."""
    # Initialize the custom CLI
    cli = ClimaXWildfireSegmentationCLI(
        run=False,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    
    # Set up model and datamodule
    model, datamodule = cli.setup_model_datamodule()
    
    # Train the model if requested
    if cli.config.do_train:
        cli.trainer.fit(model, datamodule, ckpt_path=cli.config.ckpt_path)
    
    # Determine which checkpoint to use for evaluation
    ckpt = cli.config.ckpt_path
    if cli.config.do_train:
        ckpt = "best"
    
    # Validate the model if requested
    if cli.config.do_validate:
        cli.trainer.validate(model, datamodule, ckpt_path=ckpt)
    
    # Test the model if requested
    if cli.config.do_test:
        cli.trainer.test(model, datamodule, ckpt_path=ckpt)
    
    # Generate predictions if requested
    if cli.config.do_predict:
        predictions = cli.trainer.predict(model, datamodule, ckpt_path=ckpt)
        
        # Process and save predictions
        predictions_file_name = os.path.join(
            cli.config.trainer.default_root_dir, 
            "predictions_latest.pt"
        )
        
        # Extract input, target, and prediction data
        x_fire_inputs = []  # Last channel of input contains active fire
        targets = []        # Ground truth fire masks
        predictions_prob = []  # Predicted fire probabilities
        
        for batch in predictions:
            # Extract the active fire channel from input
            x = batch[0]  # Input data
            y = batch[1]  # Target
            y_hat = batch[2]  # Predictions
            
            # Get the active fire channel (last channel of input)
            active_fire_input = x[:, -1, :, :] if x.dim() == 4 else x[:, -1]
            x_fire_inputs.append(active_fire_input)
            
            # Get target and prediction
            targets.append(y)
            predictions_prob.append(y_hat)
        
        # Concatenate results
        x_fire_combined = torch.cat(x_fire_inputs, dim=0)
        y_combined = torch.cat(targets, dim=0)
        y_hat_combined = torch.cat(predictions_prob, dim=0)
        
        # Binary predictions (after sigmoid for binary segmentation)
        if cli.config.num_classes == 1:
            y_hat_binary = torch.sigmoid(y_hat_combined) > 0.5
            all_results = torch.stack([
                x_fire_combined,                    # Input active fire
                y_combined.squeeze(1) if y_combined.dim() > 3 else y_combined,  # Ground truth
                y_hat_binary.squeeze(1) if y_hat_binary.dim() > 3 else y_hat_binary  # Binary prediction
            ], dim=0)
        else:
            # For multi-class segmentation
            y_hat_class = torch.argmax(y_hat_combined, dim=1)
            all_results = torch.stack([
                x_fire_combined,    # Input active fire
                y_combined,         # Ground truth
                y_hat_class         # Class prediction
            ], dim=0)
        
        # Save the combined tensor
        torch.save(all_results, predictions_file_name)
        print(f"Predictions saved to {predictions_file_name}")


if __name__ == "__main__":
    main()
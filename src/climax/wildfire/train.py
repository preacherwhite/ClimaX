#!/usr/bin/env python
# Training script for Wildfire ClimaX model

import os
from pytorch_lightning.cli import LightningCLI

from wildfire_datamodule import WildfireDataModule
from wildfire_module import WildfireModule


def main():
    # Initialize Lightning with the model and data modules
    cli = LightningCLI(
        model_class=WildfireModule,
        datamodule_class=WildfireDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    
    # Create output directory
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    
    # Set up model with data module info
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.prediction_range)
    cli.model.set_val_clim(cli.datamodule.get_climatology("val"))
    cli.model.set_test_clim(cli.datamodule.get_climatology("test"))
    
    # Train the model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
    # Test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
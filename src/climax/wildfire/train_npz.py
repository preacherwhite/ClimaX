#!/usr/bin/env python
# Training script for Wildfire ClimaX model adapted for step-based training approach

import os
import time
import threading
import sys
from datetime import datetime
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import Callback

from wildfire_module import WildfireModule
from wildfire_npz_datamodule import WildfireDataModule


class StuckTrainingDetectorCallback(Callback):
    """Callback to detect and handle stuck training."""
    
    def __init__(self, stuck_threshold_minutes=15, check_interval_seconds=60):
        super().__init__()
        self.stuck_threshold_seconds = stuck_threshold_minutes * 60
        self.check_interval_seconds = check_interval_seconds
        self.last_step = 0
        self.last_step_time = None
        self.monitor_thread = None
        self.stop_monitoring = False
        self.training_started = False
    
    def on_train_start(self, trainer, pl_module):
        print("INFO: Starting stuck training detector")
        self.training_started = True
        self.last_step = trainer.global_step
        self.last_step_time = time.time()
        
        # Start monitoring thread
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_progress, args=(trainer,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update last step information
        self.last_step = trainer.global_step
        self.last_step_time = time.time()
    
    def on_train_end(self, trainer, pl_module):
        print("INFO: Training ended, stopping stuck detector")
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def _monitor_progress(self, trainer):
        """Background thread to monitor training progress."""
        while not self.stop_monitoring:
            time.sleep(self.check_interval_seconds)
            
            if not self.training_started or self.last_step_time is None:
                continue
                
            # Check if training is stuck
            time_since_last_step = time.time() - self.last_step_time
            if time_since_last_step > self.stuck_threshold_seconds:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n\n----- WARNING: TRAINING APPEARS STUCK -----")
                print(f"[{current_time}] No progress for {time_since_last_step/60:.1f} minutes!")
                print(f"Last step: {self.last_step}")
                print(f"This could indicate a deadlock in the data loading pipeline.")
                print(f"Consider restarting with fewer workers or investigating data issues.")
                print(f"-----------------------------------------\n\n")
                
                # After multiple warnings, we might want to auto-terminate
                if time_since_last_step > self.stuck_threshold_seconds * 2:
                    print(f"CRITICAL: Training stuck for too long, terminating process!")
                    # Force process termination - this is drastic but better than an infinite hang
                    os._exit(1)


def main():
    # Define callbacks including the stuck detector
    stuck_detector = StuckTrainingDetectorCallback(stuck_threshold_minutes=15)
    
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
    cli.model.set_pred_range(cli.datamodule.hparams.prediction_range)
    
    # Report on valid dates filtering if used
    if hasattr(cli.datamodule.hparams, 'valid_dates_csv') and cli.datamodule.hparams.valid_dates_csv:
        valid_dates_path = cli.datamodule.hparams.valid_dates_csv
        valid_dates_count = len(cli.datamodule.valid_dates) if cli.datamodule.valid_dates else 0
        print(f"INFO: Using date filtering from CSV: {valid_dates_path}")
        print(f"INFO: Number of valid dates: {valid_dates_count}")
    else:
        print("INFO: No date filtering applied. Using all available dates.")
    
    # Handle excluded variables if present in configuration
    if hasattr(cli.datamodule.hparams, 'excluded_vars') and cli.datamodule.hparams.excluded_vars:
        excluded_vars = cli.datamodule.hparams.excluded_vars
        cli.model.set_excluded_vars(excluded_vars)
        print(f"INFO: Passing excluded variables from DataModule to Model: {excluded_vars}")
    
    # Prepare any pre-training variable exclusion setup
    if hasattr(cli.model, 'prepare_excluded_variables'):
        cli.model.prepare_excluded_variables()
    
    # Configure step-based training
    print(f"INFO: Using step-based training with max_steps={cli.trainer.max_steps}")
    
    # Ensure step-based scheduler parameters are aligned if needed
    if hasattr(cli.model.hparams, 'max_steps') and cli.model.hparams.max_steps != cli.trainer.max_steps:
        print(f"INFO: Aligning model scheduler max_steps ({cli.model.hparams.max_steps}) with trainer max_steps ({cli.trainer.max_steps})")
        cli.model.hparams.max_steps = cli.trainer.max_steps
    
    # Set trainer.limit_val_batches and limit_test_batches to 0 to skip validation and testing
    cli.trainer.limit_val_batches = 0
    cli.trainer.limit_test_batches = 0
    print("INFO: Validation and testing disabled. Model will only be trained.")
    
    # Add our custom stuck detector callback
    cli.trainer.callbacks.append(stuck_detector)
    
    try:
        # Train the model (only train, no validation)
        print(f"INFO: Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = time.time()
        
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        
        elapsed = (time.time() - start_time) / 60.0
        print(f"INFO: Training completed in {elapsed:.1f} minutes")
        
    except KeyboardInterrupt:
        print("INFO: Training interrupted by user")
        
    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nAttempting to save checkpoint before exiting...")
        try:
            if cli.trainer.global_step > 0:
                # Try to save a checkpoint even after failure
                save_path = os.path.join(cli.trainer.default_root_dir, "checkpoints", f"emergency_save_step_{cli.trainer.global_step}.ckpt")
                cli.trainer.save_checkpoint(save_path)
                print(f"Successfully saved emergency checkpoint to {save_path}")
            else:
                print("No training steps completed, no checkpoint to save.")
        except Exception as save_err:
            print(f"Failed to save emergency checkpoint: {save_err}")
        
        sys.exit(1)
    

if __name__ == "__main__":
    main()
import time

import keras_tuner as kt
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback


class TrialProgressCallback(Callback):
    def __init__(self, tuner, trial_id):
        super().__init__()
        self.tuner = tuner
        self.trial_id = trial_id
        self.trial_start_time = None
        self.epoch_history = []

    def on_train_begin(self, logs=None):
        self.trial_start_time = time.time()
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Store epoch information
        epoch_info = {
            "epoch": epoch + 1,
            "loss": logs.get("loss", 0),
            "val_loss": logs.get("val_loss", 0),
            "lr": logs.get("lr", 0),
        }
        self.epoch_history.append(epoch_info)

        # Clear output and show progress
        clear_output(wait=True)

        # Show best trial details
        self.display_best_trial_info()

        # Show only last 2 epochs
        print("\n" + "=" * 60)
        print(f"TRIAL {self.trial_id} - RECENT EPOCH PROGRESS")
        print("=" * 60)

        epochs_to_show = (
            self.epoch_history[-2:]
            if len(self.epoch_history) >= 2
            else self.epoch_history
        )

        for ep in epochs_to_show:
            print(
                f"Epoch {ep['epoch']:3d} - "
                f"Loss: {ep['loss']:.6f} - "
                f"Val Loss: {ep['val_loss']:.6f} - "
                f"LR: {ep['lr']:.2e}"
            )

        if len(self.epoch_history) > 2:
            print(f"... (showing last 2 of {len(self.epoch_history)} epochs)")

    def display_best_trial_info(self):
        print("=" * 60)
        print("BEST TRIAL SUMMARY")
        print("=" * 60)

        try:
            # Get best trial info
            best_trials = self.tuner.oracle.get_best_trials(num_trials=1)
            if best_trials:
                best_trial = best_trials[0]

                print(f"Trial ID: {best_trial.trial_id}")
                print(f"Best Validation Loss: {best_trial.score:.6f}")

                # Calculate trial duration if available
                if hasattr(best_trial, "start_time") and hasattr(
                    best_trial, "end_time"
                ):
                    if best_trial.end_time:
                        duration = best_trial.end_time - best_trial.start_time
                        print(f"Trial Duration: {duration:.2f} seconds")

                # Show hyperparameters
                print("\nBest Hyperparameters:")
                for (
                    param_name,
                    param_value,
                ) in best_trial.hyperparameters.values.items():
                    print(f"  {param_name}: {param_value}")

                # Show metrics if available
                if hasattr(best_trial, "metrics") and best_trial.metrics:
                    print("\nBest Trial Metrics:")
                    for (
                        metric_name,
                        metric_value,
                    ) in best_trial.metrics.get_last_value().items():
                        print(f"  {metric_name}: {metric_value:.6f}")

            else:
                print("No completed trials yet.")

        except Exception as e:
            print(f"Could not retrieve best trial info: {str(e)}")


class CustomTuner(kt.RandomSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_times = {}

    def run_trial(self, trial, *args, **kwargs):
        trial_start = time.time()

        # Create progress callback for this trial
        progress_callback = TrialProgressCallback(self, trial.trial_id)

        # Add progress callback to existing callbacks
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(progress_callback)
        kwargs["callbacks"] = callbacks

        try:
            result = super().run_trial(trial, *args, **kwargs)

            # Record trial completion time
            trial_end = time.time()
            trial_duration = trial_end - trial_start
            self.trial_times[trial.trial_id] = trial_duration

            # Final summary for completed trial
            clear_output(wait=True)
            print("=" * 60)
            print(f"TRIAL {trial.trial_id} COMPLETED")
            print("=" * 60)
            print(f"Final Validation Loss: {trial.score:.6f}")
            print(f"Trial Duration: {trial_duration:.2f} seconds")
            print(f"Completed Trials: {len(self.oracle.trials)}/{self.max_trials}")

            return result

        except Exception as e:
            print(f"Trial {trial.trial_id} failed: {str(e)}")
            raise
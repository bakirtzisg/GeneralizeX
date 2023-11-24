import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class StopTrainingOnSuccessRateThreshold(BaseCallback):
    """
    Stop the training once a threshold in success rate
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param success_threshold:  Minimum success rate per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 
        1 for indicating when training ended because 
        episodic success rate threshold reached
    """
    parent: EvalCallback

    def __init__(self, success_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.success_threshold = success_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnSuccessRateThreshold`` callback must be used with an ``EvalCallback``"
        success_rate = np.mean(self.parent._is_success_buffer)
        continue_training = bool(success_rate < self.success_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean success rate {success_rate:.2f} "
                f" is above the threshold {self.success_threshold}"
            )
        return continue_training
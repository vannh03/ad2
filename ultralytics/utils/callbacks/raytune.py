from ultralytics.utils import SETTINGS
try:
    assert SETTINGS['raytune'] is True
    from ray import tune
    assert hasattr(tune, 'get_context')
except (ImportError, AssertionError):
    tune = None

def on_fit_epoch_end(trainer):
    """Report training metrics to Ray Tune at epoch end when a Ray session is active.

    Captures metrics from the trainer object and sends them to Ray Tune with the current epoch number, enabling
    hyperparameter tuning optimization. Only executes when within an active Ray Tune session.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The Ultralytics trainer object containing metrics and epochs.

    Examples:
        >>> # Called automatically by the Ultralytics training loop
        >>> on_fit_epoch_end(trainer)

    References:
        Ray Tune docs: https://docs.ray.io/en/latest/tune/index.html
    """
    if tune.get_context().get_trial_id():
        tune.report({**trainer.metrics, 'epoch': trainer.epoch + 1})
callbacks = {'on_fit_epoch_end': on_fit_epoch_end} if tune else {}
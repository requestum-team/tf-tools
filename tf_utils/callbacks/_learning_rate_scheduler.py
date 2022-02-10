from typing import Optional, Dict, Callable, Tuple

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from ._utils import only_one_is_not_none, isnumber


class CustomLearningRateScheduler(Callback):
    def __init__(self,
                 scheduler_function: Optional[Callable[[int, float], float]] = None,
                 epoch_multiplier: Optional[float] = None,
                 epochs_dict: Dict[int, float] = None,
                 verbose: bool = True):

        super(CustomLearningRateScheduler, self).__init__()

        # check that only one of fields passed
        if not only_one_is_not_none(scheduler_function, epoch_multiplier, epochs_dict):
            raise LookupError('Only one of the parameters should be non-None.')

        if scheduler_function is not None:
            # validate scheduler_function
            if not callable(scheduler_function):
                raise TypeError('scheduler_function object should be callable.')

            def scheduler(epoch_num: int, old_learning_rate: float) -> Tuple[float, bool]:
                new_learning_rate: float = scheduler_function(epoch, old_learning_rate)

                if new_learning_rate != old_learning_rate:
                    return new_learning_rate, True
                else:
                    return old_learning_rate, False

            self._changing_function: Callable[[int, float], Tuple[float, bool]] = scheduler
        elif epoch_multiplier is not None:

            # validate epoch_multiplier parameter
            if not isnumber(epoch_multiplier):
                raise TypeError('')

            if epoch_multiplier <= 0:
                raise ValueError(f'epoch_multiplier should be positive, got {epoch_multiplier}')

            def scheduler(epoch_num: int, old_learning_rate: float) -> Tuple[float, bool]:
                return old_learning_rate * epoch_multiplier, True

            self._changing_function: Callable[[int, float], Tuple[float, bool]] = scheduler
        # epoch_dict is defined
        else:
            # validate epochs_dict parameter
            for epoch, learning_rate in epochs_dict.items():

                if not isnumber(epoch):
                    raise TypeError(f'One of the keys in epochs_dict is not a number, got {epoch}.')

                if epoch < 0:
                    raise ValueError(f'One of the keys in epochs_dict is lower than zero ({epoch}).')

                if not isnumber(learning_rate):
                    raise TypeError(f'One of the values in epochs_dict is not a number, got {learning_rate}.')

                if learning_rate < 0:
                    raise TypeError(f'One of the values in epochs_dict is lower than zero ({learning_rate}).')

            def scheduler(epoch_num, old_learning_rate) -> Tuple[float, bool]:
                if (new_learning_rate := epochs_dict.get(epoch_num, None)) is not None:
                    return new_learning_rate, True
                else:
                    return old_learning_rate, False

            self._changing_function: Callable[[int, float], Tuple[float, bool]] = scheduler

        self._verbose: bool = verbose

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        learning_rate: float = self.model.optimizer.learning_rate.numpy()
        new_lr, lr_is_changed = self._changing_function(epoch, learning_rate)

        if lr_is_changed:
            print(f'[CustomLearningRateScheduler]: learning rate changed from {learning_rate} to {new_lr}.')

            self.model.optimizer.learning_rate.assign(new_lr)

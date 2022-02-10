import sys
from pathlib import Path
import pickle
from typing import Union, Optional
import warnings

import numpy as np
from tensorflow.keras.callbacks import Callback

from ._utils import isnumber


class ModelSaver(Callback):

    def __init__(self,
                 folder_to_save_all: Optional[Union[str, Path]] = None,
                 monitor: str = 'val_loss',
                 each_n: Optional[Union[int, float]] = None,
                 save_last: bool = True,
                 include_optimizer: bool = False):
        super().__init__()

        if folder_to_save_all is not None:
            if not (folder_to_save_all := Path(folder_to_save_all)).exists():
                raise OSError(f'{str(folder_to_save_all)} does not exist.')

            self.__folder: Path = folder_to_save_all
        else:
            self.__folder: Path = Path(sys.argv[0]).parent.resolve()

        if each_n is not None:
            if not isnumber(each_n):
                raise TypeError('each_n value should be of type int or float.')

            if each_n % 1 != 0:
                raise ValueError(f'Invalid value for each_n parameter, it should be whole, got {each_n}')

            if each_n < 0:
                raise ValueError(f'Invalid value for each_n parameter, it should be non-negative, got {each_n}')

        self.__each_n: Optional[int] = each_n

        self.__monitor: str = monitor
        self.__best = np.inf
        self.__include_optimizer: bool = include_optimizer
        self.__save_last: bool = save_last

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:

        if self.__monitor not in logs:

            if self.__monitor != 'val_loss':
                warnings.warn('Monitor metric was not found. Try to change monitor to val_loss...')

            if 'val_loss' in logs:
                warnings.warn('Monitor was changed to val_loss.')

                self.__monitor: str = 'val_loss'
            else:
                warnings.warn('Can not change monitor to val_loss, changed to loss')

                self.__monitor: str = 'loss'

        current_metric: float = logs.get(self.__monitor)

        if current_metric < self.__best:
            save_path: Path = self.__folder / 'best'
            self.model.save_weights(save_path.with_suffix('.h5'))

            if self.__include_optimizer:
                with open(save_path.with_suffix('.opt'), 'wb') as dump_file:
                    pickle.dump(self.model.optimizer.weights, dump_file)

            self.__best: float = current_metric

        if self.__each_n is not None:
            if epoch % self.__each_n == 0:
                save_path: Path = self.__folder / str(epoch)
                self.model.save_weights(save_path.with_suffix('.h5'))

                if self.__include_optimizer:
                    with open(save_path.with_suffix('.opt'), 'wb') as dump_file:
                        pickle.dump(self.model.optimizer.weights, dump_file)

        if self.__save_last:
            # save last model after each epoch
            save_path: Path = self.__folder / 'last'
            self.model.save_weights(save_path.with_suffix('.h5'))

            if self.__include_optimizer:
                with open(save_path.with_suffix('.opt'), 'wb') as dump_file:
                    pickle.dump(self.model.optimizer.weights, dump_file)
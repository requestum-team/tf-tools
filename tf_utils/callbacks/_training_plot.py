import sys
from typing import List, Dict, Union, Optional
from pathlib import Path

import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams as rc_params


class TrainingPlot(tf.keras.callbacks.Callback):

    def __init__(self,
                 start_from: Union[int, float] = 5,
                 folder_to_save: Optional[Union[str, Path]] = None,
                 visualization_params: Optional[Dict[str, object]] = None):

        # validate start_from argument
        if isinstance(start_from, int):
            if start_from < 0:
                raise ValueError(f'Invalid value of start_from argument. Expected start_from > 0, got {start_from}.')

            self.__start_from: int = start_from
        elif isinstance(start_from, float):
            if start_from % 1 != 0:
                raise ValueError(
                    f'start_from float value can only be integer. Expected start_from % 1 = 0, got {start_from}.')
            elif start_from < 0:
                raise ValueError(f'Invalid value of start_from argument. Expected start_from > 0, got {start_from}.')

            self.__start_from: int = int(start_from)
        else:
            raise TypeError(
                f'Unexpected type of start_from argument. Expected type \'int\' or \'float\', got {type(start_from)}')

        # validate visualization_params
        if visualization_params is None:
            visualization_params: Dict[str, object] = dict()

        if not isinstance(visualization_params, dict):
            raise TypeError(f'Expected type for visualization_params is dict, got {type(visualization_params)}')

        for param_name in visualization_params:
            if param_name not in rc_params:
                raise ValueError('Unknown parameter for rcParams. Use matplotlib.pyplot.rcParams.keys(),'
                                 'to get all available parameters names.')

        self.__visualization_params: Dict[str, object] = visualization_params
        self.__has_validation: bool = False

        # validate self.__folder_to_save argument
        if folder_to_save:

            if not (folder_to_save := Path(folder_to_save)).exists():
                raise OSError(f'{str(self.__folder_to_save)} does not exist.')

            self.__folder_to_save: Path = folder_to_save
        else:
            # result will be saved to folder that contains execute file
            self.__folder_to_save: Path = Path(sys.argv[0]).parent.resolve()

        self.__metrics_values: Dict[str, List[float]] = dict()

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs):

        # skip current epoch
        if epoch >= self.__start_from:

            # we need validate metrics only once
            if epoch == self.__start_from:

                if 'val_loss' in logs:
                    self.__has_validation: bool = True

                for metric_name in logs:
                    self.__metrics_values[metric_name] = list()

            for metric_name in self.__metrics_values:
                self.__metrics_values[metric_name].append(logs.get(metric_name))

            # save old rcParams
            old_rc_params: Dict[str, object] = rc_params.copy()

            # update with users params
            rc_params.update(self.__visualization_params)

            # visualize each metric
            for metric_name in [key
                                for key in self.__metrics_values
                                if not key.startswith('val_')]:

                # clear figure after previous call
                plt.clf()

                # create chart's x axis
                x: range = range(self.__start_from, epoch + 1)

                plt.plot(x, self.__metrics_values[metric_name], label=f'{metric_name}')

                if self.__has_validation:
                    plt.plot(x, self.__metrics_values['val_' + metric_name], label=f'val_{metric_name}')

                plt.xlabel('Epoch #')
                plt.ylabel('Metric value')
                plt.title(f'{metric_name}')

                # if metric is in __metrics_filenames, filename will be __metrics_filenames[metric]
                # else filename will be {metric_name}.png
                filename: Path = (self.__folder_to_save / metric_name).with_suffix('.png')

                plt.legend()
                plt.savefig(filename)

            # back old rcParams
            rc_params.update(old_rc_params)

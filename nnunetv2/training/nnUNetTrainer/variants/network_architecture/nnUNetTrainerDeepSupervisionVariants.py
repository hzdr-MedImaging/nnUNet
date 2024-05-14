import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from scipy.stats import betabinom
import torch


def get_betabinom_weights(nstages, a, b):
    x = range(nstages)
    a = np.clip(a, 1e-10, None)
    b = np.clip(b, 1e-10, None)
    vals = betabinom.pmf(x, nstages - 1, a, b)
    return vals / sum(vals)


class nnUNetTrainer_betaBinomDS(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True
        self.ds_stages_skipped = 0

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

        deep_supervision_scales = self._get_deep_supervision_scales()
        nstages = len(deep_supervision_scales) - self.ds_stages_skipped
        progress = self.current_epoch / (self.num_epochs - 1)
        weights = get_betabinom_weights(nstages, 500 * (1 - progress), 1000 * progress)
        weights = np.append(weights, np.zeros(self.ds_stages_skipped))
        # see nnUNetTrainer for explanations
        # if self.is_ddp and not self._do_i_compile():
        #     weights[-1] = 1e-6
        # else:
        #     weights[-1] = 0

        self.loss.update_weights(weights)
        self.print_to_log_file(
            f"DS weights: {np.round(weights, decimals=2)}")


class nnUNetTrainer_betaBinomDSm1(nnUNetTrainer_betaBinomDS):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.ds_stages_skipped = 1


class nnUNetTrainer_betaBinomDSm2(nnUNetTrainer_betaBinomDS):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.ds_stages_skipped = 2



def __main__():
    def a(progress, mul=500):
        return mul * (1 - progress)

    def b(progress, mul=1000):
        return mul * progress

    def get_progress(epoch, nepoch):
        return epoch / (nepoch - 1)

    nstages = 5
    nepoch = 1000

    x = range(nstages)

    # Define initial parameters
    init_epoch = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = ax.plot(x, get_betabinom_weights(nstages, a(get_progress(init_epoch, nepoch)), b(get_progress(init_epoch, nepoch))), lw=2)
    ax.set_xlabel('Stage index')

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axepoch = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    epoch_slider = Slider(
        ax=axepoch,
        label='Epoch',
        valmin=0,
        valmax=nepoch - 1,
        valinit=init_epoch,
        valstep=1
    )

    # axbox_a = fig.add_axes([0.1, 0.05, 0.8, 0.075])
    # text_box_a = TextBox(axbox_a, "A mult", textalignment="center")
    # text_box_a.set_val("2")  # Trigger `submit` with the initial string.
    #
    # axbox_b = fig.add_axes([0.1, 0.05, 0.8, 0.075])
    # text_box_a = TextBox(axbox_b, "B mult", textalignment="center")
    # text_box_a.set_val(2)  # Trigger `submit` with the initial string.

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(get_betabinom_weights(nstages, a(get_progress(val, nepoch)), b(get_progress(val, nepoch))))
        fig.canvas.draw_idle()

    # register the update function with each slider
    epoch_slider.on_changed(update)
    # text_box_a.on_submit(update)

    plt.show()


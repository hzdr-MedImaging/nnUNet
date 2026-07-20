from time import time
from typing import List

import numpy as np
import torch
from torch import autocast, nn
from torch import distributed as dist

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss, CompositeLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import join


class nnUNetTrainerAuxEnh(nnUNetTrainer):
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # detect lesions
        target_sum = torch.sum(target[0], dim=list(range(2, target[0].ndim)))
        aux_target = (target_sum > 0.5).float()

        data = data.to(self.device, non_blocking=True)
        aux_target = aux_target.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network.forward_heads(data)
            # del data
            l = self.loss([output["seg"], target], [output["aux"], aux_target])

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # detect lesions
        target_sum = torch.sum(target[0], dim=list(range(2, target[0].ndim)))
        aux_target = (target_sum > 0.5).float()

        data = data.to(self.device, non_blocking=True)
        aux_target = aux_target.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_dict = self.network.forward_heads(data)
            output = output_dict['seg']
            aux_output = output_dict['aux']
            del data
            l = self.loss([output, target], [aux_output, aux_target])

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        aux_output = torch.sigmoid(aux_output) > 0.5
        aux_correct = (aux_output == aux_target).detach().cpu().numpy()
        aux_tp = sum(aux_correct)
        aux_total = aux_correct.size

        aux_output = torch.squeeze(aux_output, 1)
        not_detected = aux_output.logical_not() # tumor not detected by aux head
        # enforce background
        predicted_segmentation_onehot[not_detected] = 0
        predicted_segmentation_onehot[not_detected, 0] = 1
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_enh = tp.detach().cpu().numpy()
        fp_enh = fp.detach().cpu().numpy()
        fn_enh = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # see above
            tp_enh = tp_enh[1:]
            fp_enh = fp_enh[1:]
            fn_enh = fn_enh[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard, 'aux_tp': aux_tp, 'aux_total': aux_total, 'tp_enh': tp_enh, 'fp_enh': fp_enh, 'fn_enh': fn_enh}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        tp_enh = np.sum(outputs_collated['tp_enh'], 0)
        fp_enh = np.sum(outputs_collated['fp_enh'], 0)
        fn_enh = np.sum(outputs_collated['fn_enh'], 0)

        aux_tp = np.sum(outputs_collated['aux_tp'])
        aux_total = np.sum(outputs_collated['aux_total'])


        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        global_dc_enh_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_enh, fp_enh, fn_enh)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        mean_fg_dice_enh = np.nanmean(global_dc_enh_per_class)
        mean_accuracy = aux_tp / aux_total
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log_optional('mean_accuracy', mean_accuracy, self.current_epoch)
        self.logger.log_optional('mean_fg_dice_enh', mean_fg_dice_enh, self.current_epoch)
        self.logger.log_optional('dice_per_class_or_region_enh', global_dc_enh_per_class, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [str(np.round(i, decimals=4)) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('Pseudo dice enh', [str(np.round(i, decimals=4)) for i in
                                               self.logger.optional_logging['dice_per_class_or_region_enh'][-1]])
        self.print_to_log_file('Accuracy', np.round(self.logger.optional_logging['mean_accuracy'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file("Yayy! New best EMA pseudo Dice: ", np.round(self._best_ema, decimals=4))
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            seg_loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss,
                                   )
        else:
            seg_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss,
                                  )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            seg_loss = DeepSupervisionWrapper(seg_loss, weights)

        aux_loss = nn.BCEWithLogitsLoss()
        loss = CompositeLoss([seg_loss, aux_loss], weights=[1, self.configuration_manager.configuration['aux_loss_weight']])

        return loss

class nnUNetTrainerAuxEnh5epoch(nnUNetTrainerAuxEnh):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        assert self.fold == 0, "It makes absolutely no sense to specify a certain fold. Stick with 0 so that we can parse the results."
        self.disable_checkpointing = False
        self.num_epochs = 5

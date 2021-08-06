from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.nn.functional as F
import os


class RegressTrainer(TorchTrainer):
    def __init__(
            self,
            network,
            network_lr=1e-3,
            optimizer_class=optim.Adam,
            alt_buffer=None,
            obs_key='observations',
            regress_key='object_positions',
            log_pickle=True,
            pickle_log_rate=5,
            log_dir=None,
    ):
        super().__init__()
        self.network = network
        self.policy = network
        
        self._log_epoch = 0
        self.log_pickle = log_pickle
        self.pickle_log_rate = pickle_log_rate
        self.log_dir = log_dir

        self.network_optimizer = optimizer_class(
            self.network.parameters(),
            lr=network_lr,
        )

        self._optimizer_class = optimizer_class #for loading

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self._current_epoch = 0
        self._num_network_update_steps = 0
        self.discrete = False
        self.alt_buffer = alt_buffer
        self.regress_key=regress_key
        self.obs_key = obs_key

    def train_from_torch(self, batch, online=False):
        self._current_epoch += 1

        obs = batch[self.obs_key]
        orient = batch[self.regress_key]

        """Start with Regression"""
        pred = self.network(obs)
        network_loss = F.mse_loss(pred, orient)

        """
        Update networks
        """
        self._num_network_update_steps += 1
        self.network_optimizer.zero_grad()
        network_loss.backward(retain_graph=False)
        self.network_optimizer.step()

        if self.alt_buffer is not None:
            batch_alt = self.alt_buffer.random_batch(obs.shape[0])
            obs_new = ptu.from_numpy(batch_alt[self.obs_key])
            orient_new = ptu.from_numpy(batch_alt[self.regress_key])
            pred = self.network(obs_new)
            network_val_loss = F.mse_loss(pred, orient_new)
        else:
            network_val_loss = network_loss

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            if self.log_pickle and self._log_epoch % self.pickle_log_rate == 0:
                new_path = os.path.join(self.log_dir,'model_pkl')
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                torch.save({
                    'net_state_dict': self.network.state_dict(),
                    'optimizer': self.network_optimizer,
                }, os.path.join(new_path, str(self._log_epoch)+'.pt'))

            self.eval_statistics['Num network Updates'] = self._num_network_update_steps
            self.eval_statistics['Network Train Loss'] = np.mean(ptu.get_numpy(network_loss))
            self.eval_statistics['Network Val Loss'] = np.mean(ptu.get_numpy(network_val_loss))

            self._log_epoch += 1
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [self.network]
        return base_list

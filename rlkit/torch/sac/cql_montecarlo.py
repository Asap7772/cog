from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd
import matplotlib.pyplot as plt
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb

class CQLMCTrainer(TorchTrainer):
    def __init__(
            self, 
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs=2,
            gamma=1,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            log_pickle=True,
            pickle_log_rate=5,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,


            # Handling of the transfer setting
            hinge_trans=False,
            dist_diff=False,
            dist1 = 0,
            dist2 = 1,

            bottleneck= False,
            bottleneck_const=0.5,
            bottleneck_lagrange=False,
            only_bottleneck=False,
            log_dir=None,
            wand_b=True,
            variant_dict=None,
            real_data=False,
    ):
        super().__init__()
        self.gamma = gamma
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.log_dir = log_dir
        self.log_pickle=log_pickle
        self.pickle_log_rate=pickle_log_rate


        self.hinge_trans=hinge_trans
        self.dist_diff=dist_diff
        self.dist1=dist1
        self.dist2=dist2

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() 
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        
        self.with_lagrange = with_lagrange
        self.bottleneck_lagrange = bottleneck_lagrange
        if self.with_lagrange or self.bottleneck_lagrange: #TODO separate out later
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.bottleneck = bottleneck
        self.bottleneck_const = bottleneck_const
        self.only_bottleneck = only_bottleneck

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1
        self._log_epoch = 0
        
        self.num_qs = num_qs

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        # For implementation on the 
        self.discrete = False
        self.tsne = True
        

        self.wand_b = wand_b
        self.real_data = real_data
        if self.wand_b:
            if self.real_data:
                wandb.init(project='real_drawer_cql', reinit=True)
            else:
                wandb.init(project='cog_cql', reinit=True)
            wandb.run.name=log_dir.split('/')[-1]
            if variant_dict is not None:
                wandb.config.update(variant_dict)
    
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds, _ = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp, reparameterize=True, return_log_prob=True,
        )
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        rewards_mc = batch['mcrewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if self.num_qs == 1:
            q_new_actions = self.qf1(obs, new_obs_actions)
        else:
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions)[0],
                self.qf2(obs, new_obs_actions)[0],
            )

        policy_loss = (alpha*log_pi - q_new_actions).mean()

        if self._current_epoch < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()
        
        """
        QF Loss
        """
        q1_pred, q1_mc = self.qf1(obs, actions)
        if self.num_qs > 1:
            q2_pred, q2_mc = self.qf2(obs, actions)
        
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        if not self.max_q_backup:
            if self.num_qs == 1:
                target_q_values = self.target_qf1(next_obs, new_next_actions)
            else:
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions)[0],
                    self.target_qf2(next_obs, new_next_actions)[0],
                )
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        
        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        bellman_loss = self.qf_criterion(q1_pred, q_target)
        mc_loss = self.qf_criterion(q1_mc, rewards_mc)

        qf1_loss = self.qf_criterion(q1_pred, q_target) + self.gamma * self.qf_criterion(q1_mc, rewards_mc)
        if self.num_qs > 1:
            qf2_loss = self.qf_criterion(q2_pred, q_target) + self.gamma * self.qf_criterion(q2_mc, rewards_mc)

        if self.dist_diff:
            sizes = tuple(ptu.get_numpy(batch['batch_dist']).astype(int).tolist())
            split_obs = torch.split(obs,sizes)
            split_actions = torch.split(actions,sizes) 

            obs1, obs2 = split_obs[self.dist1], split_obs[self.dist2]
            actions1, actions2 = split_actions[self.dist1], split_actions[self.dist2]

            qf1_loss += (self.qf1(obs1, actions1).mean() - self.qf1(obs2,actions2).mean())**2
            qf2_loss += (self.qf2(obs1, actions1).mean() - self.qf2(obs2,actions2).mean())**2

        ## add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).cuda()
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

        if self.min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
        else:
            cat_q1 = torch.cat(
                [q1_rand, q1_pred.unsqueeze(1), q1_next_actions,
                 q1_curr_actions], 1
            )
            cat_q2 = torch.cat(
                [q2_rand, q2_pred.unsqueeze(1), q2_next_actions,
                 q2_curr_actions], 1
            )

        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        
        if self.bottleneck:
            qf1_bottleneck_sample, qf1_bottleneck_sample_log_prob, qf1_bottleneck_loss, qf1_bottleneck_mean, qf1_bottleneck_logstd, qf1_sample = self.qf1.detailed_forward(obs,actions)
            qf2_bottleneck_sample, qf2_bottleneck_sample_log_prob, qf2_bottleneck_loss, qf2_bottleneck_mean, qf2_bottleneck_logstd, qf2_sample = self.qf2.detailed_forward(obs,actions)
            
            if self.only_bottleneck:
                min_qf1_loss = qf1_bottleneck_loss.mean()
                min_qf2_loss = qf2_bottleneck_loss.mean()
            else:
                min_qf1_loss = min_qf1_loss + self.bottleneck_const * qf1_bottleneck_loss.mean()
                min_qf2_loss = min_qf2_loss + self.bottleneck_const * qf2_bottleneck_loss.mean() 
                reg_loss = qf1_bottleneck_loss.mean() + qf2_bottleneck_loss.mean()

            if self.bottleneck_lagrange:	
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0, max=1000000.0)	
                min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)
                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-reg_loss)*0.5 
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        if self.num_qs > 1:
            qf2_loss = qf2_loss + min_qf2_loss

        """
        Update networks
        """
        # Update the Q-functions iff 
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        if self.num_qs > 1:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            self.qf2_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        if self.num_qs > 1:
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            if self.log_pickle and self._log_epoch % self.pickle_log_rate == 0:
                new_path = os.path.join(self.log_dir,'model_pkl')
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                torch.save({
                    'qf1_state_dict': self.qf1.state_dict(),
                    'qf2_state_dict': self.qf2.state_dict(),
                    'targetqf1_state_dict': self.target_qf1.state_dict(),
                    'targetqf2_state_dict': self.target_qf2.state_dict(),
                    'policy_state_dict': self.policy.state_dict(),
                }, os.path.join(new_path, str(self._log_epoch)+'.pt'))

            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))

            if not self.discrete:
                self.eval_statistics['Std QF1 values'] = np.mean(ptu.get_numpy(std_q1))
                self.eval_statistics['Std QF2 values'] = np.mean(ptu.get_numpy(std_q2))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 in-distribution values',
                    ptu.get_numpy(q1_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 in-distribution values',
                    ptu.get_numpy(q2_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 random values',
                    ptu.get_numpy(q1_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 random values',
                    ptu.get_numpy(q2_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 next_actions values',
                    ptu.get_numpy(q1_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 next_actions values',
                    ptu.get_numpy(q2_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'actions', 
                    ptu.get_numpy(actions)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'rewards',
                    ptu.get_numpy(rewards)
                ))
            self.eval_statistics['Bellman Loss'] = np.mean(ptu.get_numpy(bellman_loss))
            self.eval_statistics['MC Loss'] = np.mean(ptu.get_numpy(mc_loss))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))

            if self.bottleneck:
                self.eval_statistics['QF1 Bottleneck Loss'] = np.mean(ptu.get_numpy(qf1_bottleneck_loss))
                self.eval_statistics['QF2 Bottleneck Loss'] = np.mean(ptu.get_numpy(qf2_bottleneck_loss))
                self.eval_statistics['QF1 Bottleneck Mean'] = np.mean(ptu.get_numpy(qf1_bottleneck_mean))
                self.eval_statistics['QF2 Bottleneck Mean'] = np.mean(ptu.get_numpy(qf2_bottleneck_mean))
                self.eval_statistics['QF1 Bottleneck LogStd'] = np.mean(ptu.get_numpy(qf1_bottleneck_logstd))
                self.eval_statistics['QF2 Bottleneck LogStd'] = np.mean(ptu.get_numpy(qf2_bottleneck_logstd))

                if self.tsne and self.log_dir is not None:
                    new_path = os.path.join(self.log_dir,'visualize')
                    if not os.path.isdir(new_path):
                        os.mkdir(new_path)

                    qf1_sample_npy = qf1_sample.detach().cpu().numpy()
                    qf1_bottleneck_sample_npy = qf1_bottleneck_sample.detach().cpu().numpy().squeeze()
                    rewards_npy = rewards.detach().cpu().numpy().squeeze()
                    
                    tsne = TSNE(n_components=2,perplexity=40,n_iter=300)
                    tsne_results = tsne.fit_transform(qf1_sample_npy)
                    fig = plt.figure()
                    plt.scatter(tsne_results.T[0], tsne_results.T[1], c=qf1_bottleneck_sample_npy)
                    plt.colorbar()
                    plt.title("TSNE on Bottleneck Sample on Epoch" + str(self._log_epoch))
                    plt.savefig(os.path.join(new_path,'qf_tsne_'+str(self._log_epoch)))

                    wandb.log({"TSNE (Q) on Bottleneck Sample on Epoch" + str(self._log_epoch): fig}, step=self._log_epoch)
                    plt.close()

                    fig = plt.figure()
                    plt.scatter(tsne_results.T[0], tsne_results.T[1], c=rewards_npy)
                    plt.colorbar()
                    plt.title("TSNE on Bottleneck Sample on Epoch" + str(self._log_epoch))
                    plt.savefig(os.path.join(new_path,'rew_tsne_'+str(self._log_epoch)))
                    wandb.log({"TSNE (Rew) on Bottleneck Sample on Epoch" + str(self._log_epoch): fig}, step=self._log_epoch)
                    plt.close()

                    pca = PCA(n_components=2)
                    pca.fit(qf1_sample_npy)
                    pca_results = pca.transform(qf1_sample_npy)
                    fig = plt.figure()
                    plt.scatter(pca_results.T[0], pca_results.T[1], c=qf1_bottleneck_sample_npy)
                    plt.colorbar()
                    plt.title("PCA on Bottleneck Sample on Epoch" + str(self._log_epoch))
                    plt.savefig(os.path.join(new_path,'qf_pca_'+str(self._log_epoch)))
                    wandb.log({"PCA (Q) on Bottleneck Sample on Epoch" + str(self._log_epoch): fig}, step=self._log_epoch)
                    plt.close()

                    fig = plt.figure()
                    plt.scatter(pca_results.T[0], pca_results.T[1], c=rewards_npy)
                    plt.colorbar()
                    plt.title("PCA on Bottleneck Sample on Epoch" + str(self._log_epoch))
                    plt.savefig(os.path.join(new_path,'rew_pca_'+str(self._log_epoch)))
                    wandb.log({"PCA (Rew) on Bottleneck Sample on Epoch" + str(self._log_epoch): fig}, step=self._log_epoch)
                    plt.close()

                    print(new_path)      
            
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            
            if self.with_lagrange or self.bottleneck_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                if not self.bottleneck_lagrange:
                    self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()
            
            if self.wand_b:
                wandb.log({'trainer/'+k:v for k,v in self.eval_statistics.items()}, step=self._log_epoch)
            self._log_epoch += 1
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        # return dict(
        #     policy=self.policy,
        #     qf1=self.qf1,
        #     qf2=self.qf2,
        #     target_qf1=self.target_qf1,
        #     target_qf2=self.target_qf2,
        # )
        return dict(trainer=self)
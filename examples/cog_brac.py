import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

import torch
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.brac import BRACTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
import gym

import argparse, os
import roboverse
import numpy as np

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/doodad-output/'

def process_buffer(args):
    path = '/nfs/kun1/users/asap7772/cog_data/'
    # path = '/home/stian/cog_data/'
    buffers = []    
    home = os.path.expanduser("~")
    p_data_path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/asap7772/prior_data/' 
    ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))

    if args.buffer == 0:
        ba('closed_drawer_prior.npy',y='zero')
        path = p_data_path
        ba('task_singleneut_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-25T22-52-59_9750.npy')
    elif args.buffer == 1:
        ba('closed_drawer_prior.npy',y='zero')
        ba('drawer_task.npy')
    elif args.buffer == 2:
        ba('closed_drawer_prior.npy',y='zero')
        ba('drawer_task.npy',y='noise')
    elif args.buffer == 3:
        ba('closed_drawer_prior.npy',y='noise')
        ba('drawer_task.npy',y='zero')
    elif args.buffer == 4:
        ba('closed_drawer_prior.npy',y='noise')
        ba('drawer_task.npy',y='noise')
    elif args.buffer == 5:
        ba('drawer_task.npy')
        if args.old_prior_prob > 0:
            ba('closed_drawer_prior.npy',y='zero',p=args.old_prior_prob)
        path = p_data_path
        ba('grasp_newenv_Widow250DoubleDrawerOpenGraspNeutral-v0_20K_save_all_noise_0.1_2021-03-18T01-36-52_20000.npy',y='zero')
        ba('pickplace_newenv_Widow250PickPlaceMultiObjectMultiContainerTrain-v0_20K_save_all_noise_0.1_2021-03-18T01-38-58_19500.npy',y='zero')
        ba('drawer_newenv_Widow250DoubleDrawerOpenGraspNeutral-v0_20K_save_all_noise_0.1_2021-03-18T01-37-08_19500.npy', y='zero')
    elif args.buffer == 6:
        path = p_data_path
        ba('task_multneut_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-25T22-53-21_9250.npy')
        if args.old_prior_prob > 0:
            path = '/nfs/kun1/users/asap7772/cog_data/'
            ba('closed_drawer_prior.npy',y='zero',p=args.old_prior_prob)
            ba('drawer_task.npy',y='noise')
            path = p_data_path
        ba('grasp_multneut_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-24T01-17-30_10000.npy', y='zero')
        ba('double_drawer_multneut_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-24T01-19-23_9750.npy', y='zero')
    elif args.buffer == 7:
        path = p_data_path
        ba('pick_Widow250PickTray-v0_10K_save_all_noise_0.1_2021-04-03T12-13-53_10000.npy',y='zero') #prior 
        ba('place_Widow250PlaceTray-v0_5K_save_all_noise_0.1_2021-04-03T12-14-02_4750.npy') #task
    elif args.buffer == 8:
        path = '/nfs/kun1/users/asap7772/cog_data/'
        ba('pickplace_prior.npy',y='zero') #prior 
        path = p_data_path
        ba('place_Widow250PlaceTray-v0_5K_save_all_noise_0.1_2021-04-03T12-14-02_4750.npy') #task
    elif args.buffer == 9:
        path = p_data_path
        ba('pick_Widow250PickTray-v0_10K_save_all_noise_0.1_2021-04-03T12-13-53_10000.npy',y='zero') #prior 
        path = '/nfs/kun1/users/asap7772/cog_data/'
        ba('pickplace_task.npy') #task
    elif args.buffer == 10:
        path = '/nfs/kun1/users/asap7772/cog_data/'
        ba('pickplace_prior.npy',y='zero')
        ba('pickplace_task.npy') #task
    elif args.buffer == 11:
        path  = p_data_path
        ba('coglike_prior_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-00_10000.npy', y='zero')
        ba('coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy')
    elif args.buffer == 12:
        path  = p_data_path
        ba('coglike_prior_linking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-05T11-11-02_9250.npy', y='zero')
        ba('coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy')
    elif args.buffer == 13:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero',p=args.prob)
        ba('coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy',p=args.prob)
    elif args.buffer == 14:
        path  = p_data_path
        ba('prior_reset5_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-08_10000.npy', y='zero')
        ba('task_reset5_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-17_9000.npy')
    elif args.buffer == 15:
        path  = p_data_path
        ba('prior_reset10_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-23_10000.npy', y='zero')
        ba('task_reset10_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-28_10000.npy')
    elif args.buffer == 16:
        path  = p_data_path
        ba('prior_reset100_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-35_10000.npy', y='zero')
        ba('task_reset100_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-43_10000.npy')
    elif args.buffer == 17:
        path  = p_data_path
        ba('prior_reset2_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-56-50_8000.npy',y='zero')
        ba('task_reset2_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-56-55_10000.npy')
    elif args.buffer == 18:
        path  = p_data_path
        ba('prior_reset3_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-01_10000.npy',y='zero')
        ba('task_reset3_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-10_10000.npy')
    elif args.buffer == 19:
        path  = p_data_path
        ba('prior_reset1000_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-17_9000.npy',y='zero')
        ba('task_reset1000_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-38_10000.npy')
    elif args.buffer == 20:
        path  = p_data_path
        ba('prior_reset10000_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-44_10000.npy',y='zero')
        ba('task_reset10000_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-52_9000.npy')
    elif args.buffer == 21:
        path  = p_data_path
        ba('prior_resetinf_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-59_9000.npy',y='zero')
        ba('task_resetinf_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-58-08_10000.npy')
    elif args.buffer == 22:
        ba('closed_drawer_prior.npy',p=args.prob,y='zero')
        ba('drawer_task.npy',p=args.prob)
    elif args.buffer == 23:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero')
        ba('randobj_2_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-04-15T14-05-01_10000.npy')
    elif args.buffer == 24:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero')
        ba('randobj_5_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-04-15T14-05-10_10000.npy')
    elif args.buffer == 25:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero')
        ba('randobj_10_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-04-15T14-05-18_9000.npy')
    elif args.buffer == 26:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
        ba('coglike_task_noise0.1_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.1_2021-04-23T02-22-30_4750.npy',p=args.prob,)
    elif args.buffer == 27:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
        ba('coglike_task_noise0.15_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.15_2021-04-23T02-22-39_4625.npy',p=args.prob,)
    elif args.buffer == 28:
        path  = p_data_path
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
        ba('coglike_task_noise0.2_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.2_2021-04-23T02-22-44_4875.npy',p=args.prob,)
    elif args.buffer == 28:
        ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
        ba('coglike_task_noise0.2_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.2_2021-04-23T02-22-44_4875.npy',p=args.prob,)
    elif args.buffer == 29:
        ba('pickplace_prior.npy', p=args.prob,y='zero')
        ba('pickplace_task.npy', p=args.prob)
    elif args.buffer == 30:
        path  = p_data_path
        ba('pick_10obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-26_4500.npy', p=args.prob,y='zero')
        ba('place_10obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-31_4875.npy', p=args.prob)
    elif args.buffer == 31:
        path  = p_data_path
        ba('pick_5obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-36_4750.npy', p=args.prob,y='zero')
        ba('place_5obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-39_4750.npy', p=args.prob)
    elif args.buffer == 32:
        path  = p_data_path
        ba('pick_2obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-43_5000.npy', p=args.prob,y='zero')
        ba('place_2obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-49_5000.npy', p=args.prob)
    elif args.buffer == 33:
        ba('blocked_drawer_1_prior.npy', p=args.prob,y='zero')
        ba('drawer_task.npy', p=args.prob)
    elif args.buffer == 34:
        path = ''
        if args.azure:
            ba(os.path.join(os.expand_user('~'), 'grasping35obj', 'may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy'))
            ba(os.path.join(os.expand_user('~'), 'grasping35obj', 'may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy'))
        else:
            ba('/nfs/kun1/users/avi/scripted_sim_datasets/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy')
            ba('/nfs/kun1/users/avi/scripted_sim_datasets/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy')
    elif args.buffer == 35:    
        path  = p_data_path
        ba('pick_35obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-05-07T01-17-10_4375.npy', p=args.prob, y='zero')
        ba('place_35obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-17-42_4875.npy', p=args.prob)
    elif args.buffer == 36:
        path  = p_data_path
        ba('pick_20obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-05-07T01-17-01_4625.npy', p=args.prob,
            y='zero')
        ba('place_20obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-06-14T21-53-31_5000.npy', p=args.prob)
    elif args.buffer == 37:
        path  = p_data_path
        ba('drawer_prior_multobj_Widow250DoubleDrawerOpenGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-06-23T11-52-07_10000.npy', p=args.prob, y='zero')
        ba('drawer_task_multobj_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-06-23T11-52-15_9750.npy', p=args.prob)
    elif args.buffer == 38:
        path  = p_data_path
        ba('drawer_prior_multobj_Widow250DoubleDrawerOpenGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-08-02T23-59-33_9500.npy', p=args.prob, y='zero')
        ba('drawer_task_multobj_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-08-02T23-59-38_9500.npy', p=args.prob)
    elif args.buffer == 39:
        path  = p_data_path
        ba('drawer_prior_overlap_Widow250DoubleDrawerOpenGraspNeutralRandObjOverlap-v0_10K_save_all_noise_0.1_2021-08-02T23-58-16_9500.npy', p=args.prob, y='zero')
        ba('drawer_task_overlap_Widow250DoubleDrawerGraspNeutralRandObjOverlap-v0_10K_save_all_noise_0.1_2021-08-02T23-58-16_9500.npy', p=args.prob)
    elif args.buffer == 9000:
        variant['debug'] = True
        path  = p_data_path
        ba('debug.npy',y='noise')
        ba('debug.npy',y='noise')
    elif args.buffer == 9001: #for testing wandb code
        variant['debug'] = False 
        path  = p_data_path
        ba('debug.npy',y='noise')
        ba('debug.npy',y='noise')
    return buffers

def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    if variant['num_sample'] != 0:
        eval_env.num_obj_sample=variant['num_sample']
    if False and variant['debug_scale_actions']:
        class ActionWrapper(gym.ActionWrapper):
            def __init__(self, env):
                super().__init__(env)
            
            def action(self, act):
                if variant['scale_type'] == 1:
                    act = np.concatenate((act-np.concatenate((act[:-2]*0.9,[0,0])), act))
                elif variant['scale_type'] == 2:
                    eps = np.random.uniform(-0.05, 0.05, act[:-2].shape)
                    act = np.concatenate((eps, act[:-2]-eps, act[-2:]))
                return act
        eval_env = ActionWrapper(eval_env)
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size
    print(action_dim)

    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )
    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    cnn_params.update(
        output_size=256,
        added_fc_input_size=0,
        hidden_sizes=[1024, 512],
    )

    policy_obs_processor = CNN(**cnn_params)

    behavior_policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    state_dict = torch.load(variant['behavior_path'])['policy_state_dict']
    behavior_policy.load_state_dict(state_dict)

    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    observation_key = 'image'
    replay_buffer = load_data_from_npy_chaining(
            variant,
            expl_env, 
            observation_key,
            duplicate=False,
            num_traj=variant['num_traj'],
            debug_scale_actions=False,
            debug_shift=False,
            scale_type=False,
            hist_state=False,
            num_hist=False,
        )

    # Translate 0/1 rewards to +4/+10 rewards.
    if variant['use_positive_rew']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards * 6.0
            replay_buffer._rewards = replay_buffer._rewards + 4.0
        assert set(np.unique(replay_buffer._rewards)).issubset(
            set(6.0 * np.array([0, 1]) + 4.0))

    trainer = BRACTrainer(
        env=eval_env,
        policy=policy,
        behavior_policy=behavior_policy.to(ptu.device),
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        beta=variant['beta'],
        log_dir=variant['log_dir'],
        variant_dict=variant,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=False,
        batch_rl=True,
        **variant['algorithm_kwargs']
    )
    video_func = VideoSaveFunction(variant)
    algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            num_epochs=3000,
            num_eval_steps_per_epoch=300,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=30,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--beta", default=1.0, type=float, help="Value of alpha in CQL")
    parser.add_argument("--use-positive-rew", action="store_true", default=False)
    parser.add_argument("--policy-eval-start", default=10000,type=int)
    parser.add_argument("--policy-lr", default=1e-4, type=float)
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--buffer", default=0, type=int)
    parser.add_argument('--behavior_path', default='/nfs/kun1/users/asap7772/cog/data/behavior-bc/behavior_bc_2021_08_18_21_07_43_0000--s-0/model_pkl/200.pt', type=str)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument("--prob", default=1, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--azure", action='store_true')
    parser.add_argument('--eval_num', default=0, type=int)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['num_sample'] = args.eval_num
    variant['num_traj'] = args.num_traj
    variant['prob'] = args.prob
    variant['env'] = args.env
    variant['buffer'] = args.buffer
    variant['behavior_path'] = args.behavior_path
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = \
        args.num_eval_per_epoch*args.max_path_length

    buffers = process_buffer(args)
    variant['buffer'] = buffers
    variant['bufferidx'] = args.buffer
    variant['beta'] = args.beta
    variant['behavior_path'] = args.behavior_path

    if args.buffer in [5,6]:
        variant['prior_buffer'] = buffers[1:]
        variant['task_buffer'] = buffers[0]
    else:
        variant['prior_buffer'] = buffers[0]
        variant['task_buffer'] = buffers[1]

    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['discount'] = args.discount

    # Translate 0/1 rewards to +4/+10 rewards.
    variant['use_positive_rew'] = args.use_positive_rew
    variant['seed'] = args.seed

    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-cog-{}'.format(args.env)
    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)
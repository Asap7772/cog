import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer_real import *
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.sac.cql_montecarlo import CQLMCTrainer
from rlkit.torch.sac.cql_bchead import CQLBCTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.envs.dummy_env import DummyEnv
from rlkit.launchers.launcher_util import setup_logger

import argparse, os
import roboverse
import numpy as np

DEFAULT_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
CUSTOM_LOG_DIR = '/home/asap7772/doodad-output'


def experiment(variant):
    eval_env = DummyEnv()
    if variant['num_sample'] != 0:
        eval_env.num_obj_sample=variant['num_sample']
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size
    print(action_dim)

    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=64,
        input_height=64,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )
    if variant['mcret'] or variant['bchead']:
        qf1 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'], width=64, height=64)
        qf2 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'], width=64, height=64)
        target_qf1 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'], width=64, height=64)
        target_qf2 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'], width=64, height=64)
    elif variant['bottleneck']:
        qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'], width=64, height=64)
        qf2 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'], width=64, height=64)
        target_qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'], width=64, height=64)
        target_qf2 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'], width=64, height=64)
    else:
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
    paths = []
    if args.azure:
        data_path = '/home/asap7772/drawer_data'
    else:
        data_path = '/nfs/kun1/users/ashvin/data/val_data'
    if args.buffer == 0:
        print('lid on')
        paths.append((os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_putlidon_rew.pkl')))
    elif args.buffer == 1:
        print('lid off')
        paths.append((os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_takeofflid_rew.pkl')))
    elif args.buffer == 2:
        print('tray')
        paths.append((os.path.join(data_path,'fixed_tray_demos.npy'), os.path.join(data_path,'fixed_tray_demos_rew.pkl')))
    elif args.buffer == 3:
        print('drawer')
        paths.append((os.path.join(data_path,'fixed_drawer_demos.pkl'), os.path.join(data_path,'fixed_drawer_demos_rew.pkl')))
    else:
        assert False
    
    replay_buffer = get_buffer(observation_key=observation_key)
    for path, rew_path in paths:
        load_path(path, rew_path, replay_buffer)

    if variant['val']:
        #TODO change
        print('validation')
        replay_buffer_val = None
    else:
        print('no validation')
        replay_buffer_val = None

    # Translate 0/1 rewards to +4/+10 rewards.
    if variant['use_positive_rew']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards * 6.0
            replay_buffer._rewards = replay_buffer._rewards + 4.0
        assert set(np.unique(replay_buffer._rewards)).issubset(
            set(6.0 * np.array([0, 1]) + 4.0))

    if variant['mcret']:
        trainer = CQLMCTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            only_bottleneck = variant['only_bottleneck'],
            variant_dict=variant,
            gamma=variant['gamma'],
            real_data=True,
            **variant['trainer_kwargs']
        )
    elif variant['bchead']:
        trainer = CQLBCTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            only_bottleneck = variant['only_bottleneck'],
            variant_dict=variant,
            gamma=variant['gamma'],
            real_data=True,
            **variant['trainer_kwargs']
        )
    else:
        trainer = CQLTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            only_bottleneck = variant['only_bottleneck'],
            variant_dict=variant,
            validation=variant['val'],
            validation_buffer=replay_buffer_val,
            real_data=True,
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
            num_eval_steps_per_epoch=5,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=5,
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

            # Target nets/ policy vs Q-function update
            policy_eval_start=10000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=5.0,

            # lagrange
            with_lagrange=False,  # Defaults to False
            lagrange_thresh=5.0,

            # extra params
            num_random=1,
            max_q_backup=False,
            deterministic_backup=False,
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
    parser.add_argument("--bottleneck", action='store_true')
    parser.add_argument('--bottleneck_const', type=float, default=0.5)
    parser.add_argument('--bottleneck_dim', type=int, default=16)
    parser.add_argument('--bottleneck_lagrange', action='store_true')
    parser.add_argument("--deterministic_bottleneck", action="store_true", default=False)
    parser.add_argument("--only_bottleneck", action="store_true", default=False)
    parser.add_argument("--mcret", action='store_true')
    parser.add_argument("--bchead", action='store_true')
    parser.add_argument("--azure", action='store_true', default=False)
    parser.add_argument("--prior-buffer", type=str, default=DEFAULT_PRIOR_BUFFER)
    parser.add_argument("--task-buffer", type=str, default=DEFAULT_TASK_BUFFER)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--min-q-weight", default=1.0, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--use-lagrange", action="store_true", default=False)
    parser.add_argument("--lagrange-thresh", default=5.0, type=float,
                        help="Value of tau, used with --use-lagrange")
    parser.add_argument("--use-positive-rew", action="store_true", default=False)
    parser.add_argument("--duplicate", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--max-q-backup", action="store_true", default=False,
                        help="For max_{a'} backups, set this to true")
    parser.add_argument("--no-deterministic-backup", action="store_true",
                        default=False,
                        help="By default, deterministic backup is used")
    parser.add_argument("--policy-eval-start", default=10000,
                        type=int)
    parser.add_argument("--policy-lr", default=1e-4, type=float)
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--prob", default=1, type=float)
    parser.add_argument("--old_prior_prob", default=0, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument('--eval_num', default=0, type=int)
    parser.add_argument("--name", default='test', type=str)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['val'] = args.val

    variant['prior_buffer'] = args.prior_buffer
    variant['task_buffer'] = args.task_buffer
    
    variant['bottleneck'] = args.bottleneck
    variant['bottleneck_const'] = args.bottleneck_const
    variant['bottleneck_lagrange'] = args.bottleneck_lagrange
    variant['bottleneck_dim'] = args.bottleneck_dim
    variant['deterministic_bottleneck']=args.deterministic_bottleneck
    variant['only_bottleneck'] = args.only_bottleneck
    variant['gamma'] = args.gamma
    variant['num_traj'] = args.num_traj
    variant['num_sample'] = args.eval_num
    
    variant['debug'] = False
    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)

    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = \
        not args.no_deterministic_backup
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['with_lagrange'] = args.use_lagrange
    variant['duplicate'] = args.duplicate
    variant['mcret'] = args.mcret
    variant['bchead'] = args.bchead

    # Translate 0/1 rewards to +4/+10 rewards.
    variant['use_positive_rew'] = args.use_positive_rew
    variant['seed'] = args.seed

    ptu.set_gpu_mode(True)
    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None
    
    variant['base_log_dir'] = base_log_dir
    
    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)

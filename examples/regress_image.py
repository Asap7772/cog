import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy, load_data_from_npy_mult, load_data_from_npy_split
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector, CustomMDPPathCollector_EVAL

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.regress import RegressTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger

import argparse, os
import roboverse

# '/media/avi/data/Work/github/avisingh599/minibullet/data/'
#                   'oct6_Widow250DrawerGraspNeutral-v0_20K_save_all_noise_0.1'
#                   '_2020-10-06T19-37-26_100.npy'

# DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')

DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/doodad-output/'


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = roboverse.make(variant['env'], transpose_image=True)
    action_dim = eval_env.action_space.low.size

    if variant['multi_bin']:
        eval_env.multi_tray = True
        expl_env.multi_tray = False

    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=3,
        added_fc_input_size=0,
    )
    network = ConcatCNN(**cnn_params)

    observation_key = 'image'

    replay_buffers = load_data_from_npy_split(variant, expl_env, observation_key, train_percent=variant['train_per'])

    trainer = RegressTrainer(
        env=eval_env,
        network=network,
        alt_buffer=replay_buffers[-1],
        **variant['trainer_kwargs']
    )

    eval_policy = None
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector_EVAL(
        eval_env,
        eval_policy,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffers[0],
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
        algorithm="Regress",
        version="normal",
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            num_epochs=3000,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=0,
            min_num_steps_before_training=0,
            max_path_length=30,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            network_lr=1E-4,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Widow250MultiObjectGraspTrain-v0')
    parser.add_argument("--max-path-length", type=int, default='30')
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--min-q-weight", default=1.0, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--use-lagrange", action="store_true", default=False)
    parser.add_argument("--lagrange-thresh", default=5.0, type=float,
                        help="Value of tau, used with --use-lagrange")
    parser.add_argument("--use-positive-rew", action="store_true",
                        default=False)
    parser.add_argument("--max-q-backup", action="store_true", default=False,
                        help="For max_{a'} backups, set this to true")
    parser.add_argument("--no-deterministic-backup", action="store_true",
                        default=False,
                        help="By default, deterministic backup is used")
    parser.add_argument("--policy-eval-start", default=10000,
                        type=int)
    parser.add_argument("--network-lr", default=1e-4, type=float)
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--bin_color", action="store_true", default=False)
    parser.add_argument("--multi_bin", action="store_true", default=False)
    parser.add_argument("--mixture", action="store_true", default=False)
    parser.add_argument("--transfer", action="store_true", default=False)
    parser.add_argument("--transfer_multiview", action="store_true", default=False)
    parser.add_argument("--p", default=0.2, type=float)
    parser.add_argument('--segment_type', default='fixed_other', type = str)
    parser.add_argument('--eval_multiview', default='single', type = str)
    parser.add_argument('--larger_net', action="store_true", default=False)
    parser.add_argument('--dist_diff', action="store_true", default=False)
    parser.add_argument('--train_per', default='0.9', type = float)

    args = parser.parse_args()
    variant['transfer'] = args.transfer
    variant['mixture'] = args.mixture
    variant['p'] = args.p
    variant['bin'] = args.bin_color
    variant['segment_type'] = args.segment_type
    
    variant['transfer_multiview'] = args.transfer_multiview
    variant['eval_multiview'] = args.eval_multiview
    variant['dist_diff'] = args.dist_diff
    variant['train_per'] = args.train_per

    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)

    if args.buffer == 0:
        args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/train_grasp_orient/scripted_Widow250MultiObjectGraspTrain-v0_2021-01-07T00-41-13.npy'
    elif args.buffer == 1:
        args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/val_grasp_orient/scripted_Widow250MultiObjectGraspTrain-v0_2021-01-07T00-41-30.npy'
    elif args.buffer == 2:
        args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/all_grasp_orient/scripted_Widow250MultiObjectGraspTrain-v0_2021-01-07T00-41-35.npy'

    enable_gpus(args.gpu)
    variant['env'] = args.env
    variant['buffer'] = args.buffer
    variant['trainer_kwargs']['network_lr'] = args.network_lr
    variant['multi_bin'] = args.multi_bin

    if args.larger_net:
        variant['cnn_params'] = dict(
            kernel_sizes=[3, 3, 3, 3, 3, 3],
            n_channels=[16, 16, 16, 16,16,16],
            strides=[1, 1, 1, 1, 1, 1],
            hidden_sizes=[1024, 512, 512, 256, 256],
            paddings=[1, 1, 1,1,1,1],
            pool_type='max2d',
            pool_sizes=[2, 2, 2, 2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 2,2,2,1],
            pool_paddings=[0, 0, 0,0,0,0],
            image_augmentation=True,
            image_augmentation_padding=4,
        )
    else:
        variant['cnn_params'] = dict(
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
        )

    variant['seed'] = args.seed
    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-{}'.format(args.env)

    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    experiment(variant)

from traitlets.traitlets import default
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy, load_data_from_npy_mult, load_data_from_npy_split, load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector, CustomMDPPathCollector_EVAL
from rlkit.data_management.load_buffer_real import *

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.regress import RegressTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, ConcatCNNWrapperRegress, Mlp, ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
from rlkit.envs.dummy_env import DummyEnv
import argparse, os
import roboverse
import torch

DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/doodad-output/'


TRAIN_DIRS = [
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_pepper_in_pan/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/take_can_out_of_pan/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_broccoli_in_bowl/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_corn_into_bowl/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_pear_in_bowl/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_fork_from_basket_to_tray/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/take_broccoli_out_of_pot_or_pan/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_sushi_on_plate/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/take_broccoli_out_of_pan/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_banana_on_plate/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_red_bottle_in_sink/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/take_lid_off_pot_or_pan/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_broccoli_in_pot_or_pan/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_big_spoon_from_basket_to_tray/out.npy',
]

VAL_DIRS = [
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/take_carrot_off_plate/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_detergent_in_sink/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/put_small_spoon_from_basket_to_tray/out.npy',
    '/nfs/kun1/users/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen1/take_sushi_out_of_pan/out.npy'
]


def experiment(variant):

    input_size = output_size = 6
    hidden_sizes = [256]*3
    
    if variant['transformer']:
        network = Mlp(hidden_sizes, output_size, input_size)
    else:
        network = Mlp(hidden_sizes, output_size, input_size)

    paths = []
    for p in TRAIN_DIRS:
        paths.append((p, None))

    observation_key = 'image'
    replay_buffer = get_buffer(observation_key=observation_key, color_jitter = False, action_shape=(7,))
    for path, rew_path in paths:
        print('TRAIN', path)
        load_path_kitchen(path, rew_path, replay_buffer, terminals=False,rescale=False)

    paths = []
    for p in VAL_DIRS:
        paths.append((p, None))

    observation_key = 'image'
    replay_buffer_val = get_buffer(observation_key=observation_key, color_jitter = False, action_shape=(7,))
    for path, rew_path in paths:
        print('VAL', path)
        load_path_kitchen(path, rew_path, replay_buffer_val, terminals=False,rescale=False)
    
    trainer = RegressTrainer(
        network=network,
        alt_buffer=replay_buffer_val,
        obs_key='diff_states',
        regress_key='action_fs',
        log_dir=variant['log_dir'],
        **variant['trainer_kwargs']
    )

    expl_env = eval_env = DummyEnv()
    
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
    variant = dict(
        algorithm="Regress",
        version="normal",
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=100,
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
    parser.add_argument("--env", type=str, default='')
    parser.add_argument("--max-path-length", type=int, default='30')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--network-lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--loc", default='', type=str)
    parser.add_argument('--prob', default=1, type=float)
    parser.add_argument('--train_per', default='0.9', type = float)
    parser.add_argument('--mcret', action='store_true')
    parser.add_argument('--bchead', action='store_true')
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument("--duplicate", action="store_true", default=False)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument('--dummy', action='store_false', default=True)
    parser.add_argument('--val', action='store_false', default=True)
    parser.add_argument('--transformer', action='store_true', default=False)

    args = parser.parse_args()
    variant['loc'] = args.loc
    variant['transformer'] = args.transformer
    variant['val'] = args.val
    variant['dummy'] =args.dummy
    variant['mcret'] = args.mcret
    variant['bchead'] = args.bchead
    variant['bottleneck'] = args.bottleneck
    variant['duplicate'] = args.duplicate
    variant['num_traj'] = args.num_traj
    
    
    enable_gpus(args.gpu)
    variant['env'] = args.env
    variant['trainer_kwargs']['network_lr'] = args.network_lr

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

    variant['log_dir'] = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir, snapshot_mode='gap_and_last', snapshot_gap=10,)
    experiment(variant)

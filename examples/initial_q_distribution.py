
import os
from pickle import FALSE
import numpy as np
from rlkit.data_management.load_buffer import get_buffer_size, add_data_to_buffer
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
import roboverse
import torch
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderConcatCNN, \
    ConcatBottleneckVQVAECNN, VQVAEEncoderCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)

def load_buffer():
    home = os.path.expanduser("~")
    p_data_path =  os.path.join(home, 'prior_data/')
            
    path = '/nfs/kun1/users/asap7772/cog_data/'
    buffers = []
    ba = lambda x, p=0, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
    ba('blocked_drawer_1_prior.npy', p=1,y='zero')
    ba('drawer_task.npy', p=1)


    eval_env = roboverse.make('Widow250DoubleDrawerCloseOpenGraspNeutral-v0', transpose_image=True)

    variant = dict()
    variant['prior_buffer'] = buffers[0][0]

    with open(variant['prior_buffer'], 'rb') as f:
        data_prior = np.load(f, allow_pickle=True)
    buffer_size = get_buffer_size(data_prior)

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        eval_env,
        observation_key='image',
    )

    add_data_to_buffer(data_prior, replay_buffer, initial_sd=True)
    return replay_buffer, eval_env

def load_qfunc_pol(eval_env, path):
    action_dim = eval_env.action_space.low.size
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
        spectral_norm_conv=False,
        spectral_norm_fc=False,
    )
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
        normalize_conv_activation=False
    )

    qfunc = ConcatCNN(**cnn_params)

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
        shared_encoder=False,
    )

    parameters = torch.load(path)
    qfunc.load_state_dict(parameters['qf1_state_dict'])
    policy.load_state_dict(parameters['policy_state_dict'])
    
    qfunc.to(ptu.device)
    policy.to(ptu.device)

    return qfunc, policy

if __name__ == "__main__":
    replay_buffer, eval_env = load_buffer()

    path = '/home/asap7772/asap7772/cog_stephen/cog/data/debug-pp100-minq2/debug_pp100_minq2_2021_08_27_01_06_13_0000--s-0/model_pkl'

    lst = sorted([p for p in os.listdir(path) if p.endswith('.pt')], key= lambda p: int(p.split('.')[0]))
    qvals, epochs = [], []
    for p in lst:
        epoch = int(p.split('.')[0])
        qfunc, policy = load_qfunc_pol(eval_env, os.path.join(path, p))

        batch = np_to_pytorch_batch(replay_buffer.random_batch(100))
        qval = qfunc(batch['observations'], policy(batch['observations'])[0])
        
        qval = qval.detach().cpu().numpy()
        epochs.append(epoch)
        qvals.append(qval.mean())

        print(epoch, qval.mean())

        del qfunc, policy
        torch.cuda.empty_cache()
    
    import matplotlib.pyplot as plt
    plt.plot(epochs, qvals)
    plt.show()
    plt.savefig('qvals.png')
from rlkit.samplers.rollout_functions import rollout
import rlkit.torch.pytorch_util as ptu
import argparse
import pickle
import uuid
import roboverse
import torch
import matplotlib
import matplotlib.pyplot as plt
filename = str(uuid.uuid4())
import numpy as np
from rlkit.torch.sac.policies import TanhGaussianPolicy, GaussianPolicy, MakeDeterministic


import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining,load_data_from_npy_chaining_mult
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderConcatCNN, \
    ConcatBottleneckVQVAECNN, VQVAEEncoderCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
import gym

import argparse, os
import roboverse
import numpy as np

import os
from os.path import expanduser
def simulate_policy(args):
    env = roboverse.make(args.env, transpose_image=True)
    action_dim = env.action_space.low.size

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
        image_augmentation_padding=4)


    if args.deeper_net:
        print('deeper conv net')
        cnn_params.update(
            kernel_sizes=[3, 3, 3, 3, 3],
            n_channels=[32, 32, 32, 32, 32],
            strides=[1, 1, 1, 1, 1],
            paddings=[1, 1, 1, 1, 1],
            pool_sizes=[2, 2, 1, 1, 1],
            pool_strides=[2, 2, 1, 1, 1],
            pool_paddings=[0, 0, 0, 0, 0]
        )
    
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=9 if args.history else 3,
        output_size=1,
        added_fc_input_size=action_dim,
    )
    
    cnn_params.update(
        output_size=256,
        added_fc_input_size=args.statedim if args.imgstate else 0,
        hidden_sizes=[1024, 512],
    ) 

    print(cnn_params)

    if args.vqvae_enc:
        policy_obs_processor = VQVAEEncoderCNN(**cnn_params, num_res=6 if args.deeper_net else 3)
    else:
        policy_obs_processor = CNN(**cnn_params)

    policy_class = GaussianPolicy if args.gaussian_policy else TanhGaussianPolicy
    policy = policy_class(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    parameters = torch.load(args.policy_path)
    policy.load_state_dict(parameters['policy_state_dict'])
    
    print("Policy loaded")

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.gpu:
        ptu.set_gpu_mode(True)
    if hasattr(policy, "to"):
        policy.to(ptu.device)
    if hasattr(env, "vae"):
        env.vae.to(ptu.device)

    if args.deterministic:
        policy = MakeDeterministic(policy)

    if args.pause:
        import ipdb; ipdb.set_trace()
    policy.train(False)
    
    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean
    import torchvision.transforms.functional as F
    from PIL import Image
    import time

    def plot_img(obs_img):
        matplotlib.use('TkAgg')
        plt.figure()
        if type(obs_img) == torch.Tensor:
            from torchvision import transforms
            im_new = transforms.ToPILImage()(obs_img)
        else:
            im_new = obs_img
        plt.imshow(im_new)
        plt.show()
    
    def plot_img_mult(obs_img, num=3, channels=3):
        matplotlib.use('TkAgg')
        plt.figure()
        for i in range(num):
            plt.subplot(1,num,i+1)
            curr_img = obs_img[channels*i:channels*(i+1)]
            if type(curr_img) == torch.Tensor:
                from torchvision import transforms
                im_new = transforms.ToPILImage()(curr_img)
            else:
                im_new = curr_img
            plt.imshow(im_new)
        plt.show()

    paths = []
    for i in range(args.N):
        print('traj', i)
        next_observations = []
        observations = []
        cropped_images = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        observation = env.reset()
        prev_context = ptu.zeros(6).float()

        prev_imgs = [np.zeros_like(observation['image']) for _ in range(args.num_prev)]

        if args.history: 
            delayed_imgs = []
            for x in range(args.num_prev):
                delayed_imgs.append(observation['image'])
                action = np.concatenate((np.random.uniform(0, 0, (6,)),np.zeros((2,))))
                observation, reward, done, info = env.step(action)
            all_imgs = prev_imgs + delayed_imgs

        for j in range(args.H):
            print('trans', j)
            obs = observation['image']

            if args.history:
                curr_imgs = all_imgs[:args.num_prev+1]
                all_imgs.append(obs)
                all_imgs.pop(0)

                curr_imgs = [torch.from_numpy(x.reshape(3,48,48)) for x in curr_imgs]
                obs_img = torch.cat(tuple(curr_imgs))
            else:
                obs_img = torch.from_numpy(obs).reshape(3,48,48)

            if args.save_img:
                if args.history:
                    plot_img_mult(obs_img,args.num_prev+1)
                else:
                    plot_img(obs_img)

            if args.debug:
                action = np.random.rand(8)
            else:
                if args.scale:
                    if args.ptype == 1:
                        context = prev_context
                    elif args.ptype == 2:
                        context = ptu.from_numpy(np.random.uniform(-0.05, 0.05, (6,)))
                    else:
                        assert False
                    action = policy.forward(obs_img.flatten()[None],extra_fc_input=context if args.context else None)[0].squeeze().detach().cpu().numpy()
                    if args.ptype == 1:
                        prev_context = ptu.from_numpy(action[:6])*(1-1/1.1)
                        action = (1/1.1)* action
                    elif args.ptype == 2:
                        action = action - context #not right
                    else:
                        assert False
                else:
                    action = policy.forward(obs_img.flatten()[None],extra_fc_input=torch.from_numpy(observation['state_observation'])[None].float() if args.context else None)[0].squeeze().detach().cpu().numpy()

            print('action', action)
            old_obs = observation
            observation, reward, done, info = env.step(action)

            observations.append(old_obs)
            next_observations.append(observation)
            cropped_images.append(obs_img)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        paths.append(dict(observations=observations,next_observations=next_observations,cropped_images=cropped_images,actions=actions, rewards = rewards, dones=dones, infos=infos))
        print('saved', args.out_path)
        np.save(args.out_path, paths)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Widow250DoubleDrawerOpenGraspNeutral-v0')
    parser.add_argument('--N', type=int, default=10, help='Number of Trajectories')
    parser.add_argument('--H', type=int, default=50, help='Max length of rollout')
    parser.add_argument('--num_prev', type=int, default=2)
    # parser.add_argument()
    # parser.add_argument('--policy_path', type=str, default='/nfs/kun1/users/asap7772/cog/data/updatedbuffer-rebuttal-v1-drawer-minq2/updatedbuffer_rebuttal_v1_drawer_minq2_2021_08_22_10_39_45_0000--s-0/model_pkl/190.pt')
    parser.add_argument('--policy_path', type=str, default='/nfs/kun1/users/asap7772/cog/data/shifted-relaunchedv2-brac-drawer-beta5/shifted_relaunchedv2_brac_drawer_beta5_2021_08_20_00_41_51_0000--s-0/model_pkl/620.pt')
    
    parser.add_argument('--out_path', type=str, default='evaluation')
    parser.add_argument('--env_type', type=str, default='evaluation')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--ptype', type=int, default=1)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--gaussian_policy', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--log_diagnostics', action='store_true')
    parser.add_argument('--smdim', action='store_true')
    parser.add_argument('--vqvae_enc', action='store_true')
    parser.add_argument('--deeper_net', action='store_true')
    parser.add_argument('--imgstate', action='store_true')
    parser.add_argument('--pickle', action='store_true')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--statedim', type=int, default=3)
    args = parser.parse_args()
    simulate_policy(args)

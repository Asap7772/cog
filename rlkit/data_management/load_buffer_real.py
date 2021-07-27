import numpy as np
from numpy.core.fromnumeric import trace
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from rlkit.data_management.combined_replay_buffer_prop import CombinedReplayBuffer2
import os
from rlkit.envs.dummy_env import DummyEnv
import pickle
import matplotlib.pyplot as plt

# TODO Clean up this file
MAX_SIZE = int(1E5)
def get_buffer(observation_key='image', buffer_size=MAX_SIZE, image_shape=(64,64,3), state_shape=(3,), action_shape=(4,), imgstate=False, color_jitter=False, num_viewpoints=1):
    expl_env = DummyEnv(image_shape=image_shape,state_shape=state_shape, action_shape=action_shape)
    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
        internal_keys=None,
        color_jitter=color_jitter,
        num_viewpoints=num_viewpoints,
        jit_percent=0.9, #TODO Change
    )
    return replay_buffer

import torch

#Stephen commented on 6-10 because it was preventing headless rendering in bottleneck setup.
# import matplotlib.pyplot as plt
# def plot_img(obs_img):
#     plt.figure()
#     if type(obs_img) == torch.Tensor:
#         im_new = transforms.ToPILImage()(obs_img)
#     else:
#         im_new = obs_img
#     plt.imshow(im_new)
#     plt.savefig('/nfs/kun1/users/asap7772/cog/a.png')
#     plt.show()

from torchvision import transforms
from skimage.transform import resize

def resize_small(img):
    if img.shape[0] == 6912 or img.flatten().shape[0] == 921600:
        return img
    img = img.reshape(3,64,64)
    img = img.transpose(1,2,0)
    img = resize(img, (48, 48), anti_aliasing=True)
    img = img.transpose(2,0,1).flatten()
    return img

def parse_traj(data, des_per, num_traj):
    succ, fail = [], []
    for i in range(len(data)):
        if sum(data[i]['rewards']) > 0:
            succ.append(data[i])
        else:
            fail.append(data[i])

    num_succ = int(des_per*num_traj)
    num_fail = num_traj - num_succ
    print('succ/fail', num_succ, num_fail)
    return succ[:num_succ] + fail[:num_fail]


def load_data(path, rew_path, small_img=False, bc=False, des_per=-1, num_traj = 100):
    data = np.load(path,allow_pickle=True)
    if rew_path is not None:
        rew = pickle.load(open(rew_path, 'rb'))
        assert len(data) == len(rew)
        for i in range(len(data)):
            data[i]['rewards'] = np.array(rew[i]).tolist()

    if des_per > 0:
        data = parse_traj(data, des_per, num_traj)

    if bc:
        data = [data[i] for i in range(len(data)) if sum(data[i]['rewards']) > 0]
        print('kept', len(data), 'traj')

    if small_img:
        for i in range(len(data)):
            for j in range(len(data[i]['observations'])):
                data[i]['observations'][j]['image_observation'] = resize_small(data[i]['observations'][j]['image_observation'])
                data[i]['next_observations'][j]['image_observation'] = resize_small(data[i]['next_observations'][j]['image_observation'])
    return data

def load_path(path, rew_path, replay_buffer, small_img=False, bc=False, imgstate=False, des_per=-1, num_traj = 100):
    data = np.load(path,allow_pickle=True)
    if rew_path is not None:
        rew = pickle.load(open(rew_path, 'rb'))
        assert len(data) == len(rew)
        for i in range(len(data)):
            data[i]['rewards'] = np.array(rew[i]).tolist()

    if des_per > 0:
        data = parse_traj(data, des_per, num_traj)

    if bc:
        data = [data[i] for i in range(len(data)) if sum(data[i]['rewards']) > 0]
        print('kept', len(data), 'traj')

    if small_img:
        for i in range(len(data)):
            for j in range(len(data[i]['observations'])):
                data[i]['observations'][j]['image_observation'] = resize_small(data[i]['observations'][j]['image_observation'])
                data[i]['next_observations'][j]['image_observation'] = resize_small(data[i]['next_observations'][j]['image_observation'])
    add_data_to_buffer(data, replay_buffer, small_img=small_img, imgstate=imgstate)


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions

def add_data_to_buffer(data, replay_buffer, scale_rew=False, scale=200, shift=1, drop_last=True, small_img=False, imgstate=False):
    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        if 'next_actions' not in data[j]:
            data[j]['next_actions'] = np.concatenate((data[j]['actions'][1:], data[j]['actions'][-1:]))
        rew = np.array(data[j]['rewards']).squeeze() * (0.99**np.arange(len(data[j]['rewards'])))
        mcrewards = np.cumsum(rew[::-1])[::-1].tolist() 

        if 'latents' in data[j]:
            path = dict(
                rewards=[np.asarray([r]) for r in data[j]['rewards']],
                mcrewards=[np.asarray([r]) for r in mcrewards],
                actions=data[j]['actions'],
                next_actions =data[j]['next_actions'],
                terminals=[np.asarray([t]) for t in data[j]['terminals']],
                observations=process_images(data[j]['observations'], small_img=small_img, imgstate=imgstate),
                next_observations=process_images(data[j]['next_observations'], small_img=small_img, imgstate=imgstate),
                latents = data[j]['latents'],
                next_latents = data[j]['next_latents'],
            )
        else:
            path = dict(
                rewards=[np.asarray([r]) for r in data[j]['rewards']],
                mcrewards=[np.asarray([r]) for r in mcrewards],
                actions=data[j]['actions'],
                next_actions =data[j]['next_actions'],
                terminals=[np.asarray([t]) for t in data[j]['terminals']],
                observations=process_images(data[j]['observations'], small_img=small_img, imgstate=imgstate),
                next_observations=process_images(data[j]['next_observations'], small_img=small_img, imgstate=imgstate),
            )


        if drop_last:
            for x in path:
                #drop last transition since image size is larger
                path[x] = path[x][:-1]

        if scale_rew:
            path['rewards'] = [np.asarray([r*scale + shift]) for r in data[j]['rewards']]
        replay_buffer.add_path(path)

def process_images(observations, small_img=False, imgstate=False):
    key = '_observation' if 'image_observation' in observations[0] else ''
    output = []
    for i in range(len(observations)):
        try:
            if small_img:
                image = observations[i]['image' + key].reshape(3,48,48)
            else:
                image = observations[i]['image' + key].reshape(3,64,64)
            state = observations[i]['state' + key]
        except:
            if small_img:
                image = observations[i-1]['image' + key].reshape(3,48,48)
            else:
                image = observations[i-1]['image' + key].reshape(3,64,64)
            state = observations[i-1]['state' + key]
        if len(image.shape) == 3:
            image = image.flatten()
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(state=state, image=image) if imgstate else dict(image=image))     
    return output


def load_path_kitchen(path, rew_path, replay_buffer, rescale=True, terminals=False):
    data = np.load(path,allow_pickle=True)
    if rew_path is not None:
        rew = np.load(rew_path,allow_pickle=True)
        assert len(data) == len(rew)
        for i in range(len(data)):
            data[i]['rewards'] = np.array(rew[i]).tolist()
        
        if terminals:
            data[i]['terminals'] = np.zeros_like(data[i]['rewards'])
            data[i]['terminals'][-1] = 1
            data[i]['rewards'] = data[i]['terminals'] * 10

    add_data_to_buffer_kitchen(data, replay_buffer)

    if rescale:
        #rescaling of actions
        all_actions = replay_buffer._actions[:replay_buffer._top]
        replay_buffer._actions[:replay_buffer._top][:,0] = np.clip(all_actions[:,0]*20, -1, 1)
        replay_buffer._actions[:replay_buffer._top][:,1] = np.clip(all_actions[:,1]*20, -1, 1)
        replay_buffer._actions[:replay_buffer._top][:,2] = np.clip(all_actions[:,2]*20, -1, 1)
        replay_buffer._actions[:replay_buffer._top][:,3] = np.clip(all_actions[:,3]*20, -1, 1)
        replay_buffer._actions[:replay_buffer._top][:,4] = np.clip(all_actions[:,4]*20, -1, 1)
        replay_buffer._actions[:replay_buffer._top][:,5] = np.clip(all_actions[:,5]*20, -1, 1)
        replay_buffer._actions[:replay_buffer._top][:,6] = np.clip(all_actions[:,6]*-0.8+.9 + np.random.uniform(-.1,.1, all_actions[:,6].shape), -1, 1)
        
        replay_buffer._curr_diff[:replay_buffer._top] = np.clip(replay_buffer._curr_diff[:replay_buffer._top]*20, -1, 1)
        
        all_actions = replay_buffer._actions[:replay_buffer._top]

def add_data_to_buffer_kitchen(data, replay_buffer, num_hist = 2):
    for j in range(len(data)):
        if 'next_actions' not in data[j]:
            data[j]['next_actions'] = np.concatenate((data[j]['actions'][1:], data[j]['actions'][-1:]))
        rew = np.array(data[j]['rewards']).squeeze() * (0.99**np.arange(len(data[j]['rewards'])))
        mcrewards = np.cumsum(rew[::-1])[::-1].tolist() 


        prev_observations = []
        fv = [x['images0']*255 for x in data[j]['observations']]
        for i in range(num_hist):
            obs_prev = [np.zeros_like(fv[0])] * (i+1) + fv[:(-i-1)]
            prev_observations.append(obs_prev)
        prev_observations = prev_observations[::-1]

        viewpoints = []
        next_viewpoints = []
        if replay_buffer.num_viewpoints >= 1:
            for i in range(1,replay_buffer.num_viewpoints):
                viewpoints.append([x['images' + str(i)]*255 for x in data[j]['observations']])
                next_viewpoints.append([x['images' + str(i)]*255 for x in data[j]['next_observations']])
        
        diff = []
        for i in range(len(data[j]['observations'])):
            s1,s2 = data[j]['observations'][i]['state'], data[j]['next_observations'][i]['state']
            curr_diff = (s2-s1)-data[j]['actions'][i]
            diff.append(curr_diff[:6])

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            mcrewards=[np.asarray([r]) for r in mcrewards],
            actions=data[j]['actions'],
            next_actions=data[j]['next_actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_images_kitchen(data[j]['observations']),
            next_observations=process_images_kitchen(data[j]['next_observations']),
            viewpoints=viewpoints,
            next_viewpoints=next_viewpoints,
            curr_diff=diff,
            prev_observations=prev_observations,
        )
        replay_buffer.add_path(path)

def process_images_kitchen(observations):
    def plot_img(obs_img):
        if type(obs_img) == torch.Tensor:
            from torchvision import transforms
            im_new = transforms.ToPILImage()(obs_img.cpu())
        else:
            im_new = obs_img
        plt.imshow(im_new)

    output = []
    for i in range(len(observations)):
        image = observations[i]['images0'].reshape(3,64,64)
        # plot_img(torch.from_numpy(image))
        # plt.show()

        state = observations[i-1]['state']
        if len(image.shape) == 3:
            image = image.flatten()
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))     
    return output

if __name__ == "__main__":
    args = lambda:0 #RANDOM Object
    paths = []
    observation_key = 'image'
    
    paths.append(('/home/asap7772/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen2_room8052/put_potato_on_plate/out.npy','/home/asap7772/asap7772/real_data_kitchen/bridge_data_numpy/toykitchen2_room8052/put_potato_on_plate/out_rew.npy'))

    replay_buffer = get_buffer(observation_key=observation_key, color_jitter = True, num_viewpoints=5, action_shape=(7,))
    for path, rew_path in paths:
        print(path)
        load_path_kitchen(path, rew_path, replay_buffer)
    batch = replay_buffer.random_batch(5)
    import ipdb; ipdb.set_trace()

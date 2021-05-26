import numpy as np
from numpy.core.fromnumeric import trace
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from rlkit.data_management.combined_replay_buffer_prop import CombinedReplayBuffer2
import os
from rlkit.envs.dummy_env import DummyEnv
import pickle

# TODO Clean up this file
MAX_SIZE = int(1E5)
def get_buffer(observation_key='image', buffer_size=MAX_SIZE):
    expl_env = DummyEnv()
    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
        internal_keys=None
    )
    return replay_buffer


def load_path(path, rew_path, replay_buffer):
    data = np.load(path,allow_pickle=True)
    if rew_path is not None:
        rew = pickle.load(open(rew_path, 'rb'))
        assert len(data) == len(rew)
        for i in range(len(data)):
            data[i]['rewards'] = np.array(rew[i]).tolist()
    add_data_to_buffer(data, replay_buffer)

def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions

def add_data_to_buffer(data, replay_buffer, scale_rew=False, scale=200, shift=1, drop_last=True):
    for j in range(len(data)):
        # import ipdb; ipdb.set_trace()
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        if 'next_actions' not in data[j]:
            data[j]['next_actions'] = np.concatenate((data[j]['actions'][1:], data[j]['actions'][-1:]))
        
        rew = np.array(data[j]['rewards']) * (0.99**np.arange(len(data[j]['rewards'])))
        mcrewards = np.cumsum(rew[::-1])[::-1].tolist() 
        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            mcrewards=[np.asarray([r]) for r in mcrewards],
            actions=data[j]['actions'],
            next_actions =data[j]['next_actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_images(data[j]['observations']),
            next_observations=process_images(data[j]['next_observations'])
        )

        if drop_last:
            for x in path:
                #drop last transition since image size is larger
                path[x] = path[x][:-1]

        if scale_rew:
            path['rewards'] = [np.asarray([r*scale + shift]) for r in data[j]['rewards']]
            
        replay_buffer.add_path(path)

def process_images(observations):
    output = []
    for i in range(len(observations)):
        try:
            image = observations[i]['image_observation'].reshape(3,64,64)
        except:
            image = observations[i-1]['image_observation'].reshape(3,64,64)
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
    data_path = '/nfs/kun1/users/ashvin/data/val_data'
    paths.append( (os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_putlidon_rew.pkl')))
    
    replay_buffer = get_buffer()
    for path, rew_path in paths:
        load_path(path, rew_path, replay_buffer)
    import ipdb; ipdb.set_trace()
    
import numpy as np
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from rlkit.data_management.combined_replay_buffer_prop import CombinedReplayBuffer2

# TODO Clean up this file


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions


def add_data_to_buffer(data, replay_buffer, scale_rew=False, scale=200, shift=1):

    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_images(data[j]['observations']),
            next_observations=process_images(
                data[j]['next_observations']),
        )

        if scale_rew:
            path['rewards'] = [np.asarray([r*scale + shift]) for r in data[j]['rewards']]
            
        replay_buffer.add_path(path)


def load_data_from_npy_drawermult(variant, expl_env, observation_key, extra_buffer_size=100):
    paths = variant['buffer']
    buffers = []
    ps = []
    for p, p_params in paths:
        print(p, p_params)
        with open(p, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        num_transitions = get_buffer_size(data)
        buffer_size = num_transitions + extra_buffer_size

        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
        )
        add_data_to_buffer(data, replay_buffer)

        if p_params['alter_type'] == None:
            print('not changed')
        elif p_params['alter_type'] == 'noise':
            print('noise')
            noise = np.random.normal(0, 1, replay_buffer._rewards.shape)
            replay_buffer._rewards = replay_buffer._rewards+noise
        elif p_params['alter_type'] == 'zero':
            print('zero')
            replay_buffer._rewards = np.zeros_like(replay_buffer._rewards)
        else:
            assert False

        buffers.append(replay_buffer)
        ps.append(p_params['p'])
        print('Data loaded from npy file', replay_buffer._top)

    print('TRANSFER:', len(buffers))
    return CombinedReplayBuffer2(buffers=buffers, p=ps)


def load_data_from_npy(variant, expl_env, observation_key, extra_buffer_size=100, bin_change=False, target_segment = 'fixed_other', scale_rew=False, internal_keys=None, debug=False):
    with open(variant['buffer'], 'rb') as f:
        data = np.load(f, allow_pickle=True)
    
    if debug:
        scale = 0.01
        data = data[:int(len(data)*scale)]

    num_transitions = get_buffer_size(data)
    buffer_size = num_transitions + extra_buffer_size

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
        color_segment=bin_change,
        target_segment= target_segment,
        internal_keys=internal_keys
    )
    add_data_to_buffer(data, replay_buffer, scale_rew=scale_rew)
    print('Data loaded from npy file', replay_buffer._top)
    return replay_buffer

def load_data_from_npy_split(variant, expl_env, observation_key, extra_buffer_size=100, bin_change=False, target_segment = 'fixed_other', scale_rew=False, train_percent=0.7, debug=False):
    with open(variant['buffer'], 'rb') as f:
        data = np.load(f, allow_pickle=True)
    
    n_train = int(train_percent*len(data))
    
    data, data_v2 = data[:n_train], data[n_train:]
    
    if debug:
        scale = 0.01
        data = data[:int(len(data)*scale)]
        data_v2 = data_v2[:int(len(data_v2)*scale)]
    
    num_transitions = get_buffer_size(data)
    buffer_size = num_transitions + extra_buffer_size

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
        color_segment=bin_change,
        target_segment= target_segment,
        internal_keys=['camera_orientation']
    )
    add_data_to_buffer(data, replay_buffer, scale_rew=scale_rew)
    print('Data loaded from npy file', replay_buffer._top)

    num_transitions = get_buffer_size(data_v2)
    buffer_size = num_transitions + extra_buffer_size

    replay_buffer2 = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
        color_segment=bin_change,
        target_segment= target_segment,
        internal_keys=['camera_orientation']
    )
    add_data_to_buffer(data_v2, replay_buffer2, scale_rew=scale_rew)
    print('Data loaded from npy file', replay_buffer2._top)

    return replay_buffer, replay_buffer2

def load_data_from_npy_split_v2(variant, expl_env, observation_key, extra_buffer_size=100, bin_changes=[False]*5, target_segments = ['fixed_other']*5, p = 0.3, scale_rew=False):
    buff_locs = variant['buffer']
    buffers = []
    for i in range(len(buff_locs)):
        buff_loc = buff_locs[i]
        bin_change = bin_changes[i]
        target_segment = target_segments[i]
        
        with open(buff_loc, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        num_transitions = get_buffer_size(data)
        buffer_size = num_transitions + extra_buffer_size

        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
            color_segment=bin_change,
            target_segment= target_segment,
            internal_keys=['camera_orientation'],
        )
        add_data_to_buffer(data, replay_buffer, scale_rew=scale_rew)
        print('Data loaded from npy file', replay_buffer._top)
        buffers.append(replay_buffer)
    print('TRANSFER:', len(buffers))
    return buffers

def load_data_from_npy_mult(variant, expl_env, observation_key, extra_buffer_size=100, bin_changes=[False]*5, target_segments = ['fixed_other']*5, p = 0.3, scale_rew=False):
    buff_locs = variant['buffer']
    buffers = []
    for i in range(len(buff_locs)):
        buff_loc = buff_locs[i]
        bin_change = bin_changes[i]
        target_segment = target_segments[i]
        
        with open(buff_loc, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        num_transitions = get_buffer_size(data)
        buffer_size = num_transitions + extra_buffer_size

        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
            color_segment=bin_change,
            target_segment= target_segment
        )
        add_data_to_buffer(data, replay_buffer, scale_rew=scale_rew)
        print('Data loaded from npy file', replay_buffer._top)
        buffers.append(replay_buffer)
    print('TRANSFER:', len(buffers))
    return CombinedReplayBuffer2(buffers=buffers, p=p)

def load_data_from_npy_chaining(variant, expl_env, observation_key,
                                extra_buffer_size=100, duplicate=False):
    if type(variant['prior_buffer']) == tuple:
        variant['prior_buffer'], p_dict = variant['prior_buffer']
        variant['task_buffer'], t_dict = variant['task_buffer'] # may change
        print(p_dict)
    else:
        assert False
        
    with open(variant['prior_buffer'], 'rb') as f:
        data_prior = np.load(f, allow_pickle=True)
    with open(variant['task_buffer'], 'rb') as f:
        data_task = np.load(f, allow_pickle=True)

    buffer_size = get_buffer_size(data_prior)
    buffer_size += get_buffer_size(data_task)
    if duplicate:
        buffer_size += get_buffer_size(data_task)
    buffer_size += extra_buffer_size

    # TODO Clean this up
    if 'biased_sampling' in variant:
        if variant['biased_sampling']:
            bias_point = buffer_size - extra_buffer_size
            print('Setting bias point', bias_point)
            replay_buffer = ObsDictReplayBuffer(
                buffer_size,
                expl_env,
                observation_key=observation_key,
                biased_sampling=True,
                bias_point=bias_point,
                before_bias_point_probability=0.5,
            )
        else:
            replay_buffer = ObsDictReplayBuffer(
                buffer_size,
                expl_env,
                observation_key=observation_key,
            )
    else:
        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
        )

    add_data_to_buffer(data_prior, replay_buffer)
    top = replay_buffer._top
    print('Prior data loaded from npy file', top)
    if p_dict['alter_type'] == 'zero':
        replay_buffer._rewards[:top] = 0.0*replay_buffer._rewards[:top]
        print('Zero-ed the rewards for prior data', top)
    elif p_dict['alter_type'] == 'noise':
        noise = np.random.normal(0, 1, replay_buffer._rewards[:top].shape)
        replay_buffer._rewards[:top] = replay_buffer._rewards[:top] + noise
        print('Noise-ed the rewards for prior data', top)
    else:
        assert False

    top2 = replay_buffer._top
    add_data_to_buffer(data_task, replay_buffer)
    if t_dict['alter_type'] == 'zero':
        replay_buffer._rewards[top:top2] = 0.0*replay_buffer._rewards[top:top2]
        print('Zero-ed the rewards for task data', top)
    elif t_dict['alter_type'] == 'noise':
        noise = np.random.normal(0, 1, replay_buffer._rewards[top:top2].shape)
        replay_buffer._rewards[top:top2] = replay_buffer._rewards[top:top2] + noise
        print('Noise-ed the rewards for task data', top)
    elif t_dict['alter_type'] == None:
        pass
    else:
        assert False
    print('Task data loaded from npy file', replay_buffer._top)
    
    if duplicate:
        add_data_to_buffer(data_task, replay_buffer)
        print('Duplicate data loaded from npy file', replay_buffer._top)
    return replay_buffer

def load_data_from_npy_chaining_mult(variant, expl_env, observation_key, extra_buffer_size=100):
    variant['task_buffer'], t_dict = variant['task_buffer'] # may change
    
    data_prior = list()
    for p_buff, p_dict in variant['prior_buffer']:
        with open(p_buff, 'rb') as f:
            data_prior.append(np.load(f, allow_pickle=True)) 
    with open(variant['task_buffer'], 'rb') as f:
        data_task = np.load(f, allow_pickle=True)
    
    buffer_size = sum([get_buffer_size(x) for x in data_prior])
    buffer_size += get_buffer_size(data_task)
    buffer_size += extra_buffer_size

    if 'biased_sampling' in variant:
        if variant['biased_sampling']:
            bias_point = buffer_size - extra_buffer_size
            print('Setting bias point', bias_point)
            replay_buffer = ObsDictReplayBuffer(
                buffer_size,
                expl_env,
                observation_key=observation_key,
                biased_sampling=True,
                bias_point=bias_point,
                before_bias_point_probability=0.5,
            )
        else:
            replay_buffer = ObsDictReplayBuffer(
                buffer_size,
                expl_env,
                observation_key=observation_key,
            )
    else:
        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
        )
    
    top_prev = 0
    for buff in data_prior:
        print('prior buff', top_prev)
        add_data_to_buffer(buff, replay_buffer)
        replay_buffer._rewards[top_prev:replay_buffer._top] = 0.0
        top_prev = replay_buffer._top
    add_data_to_buffer(data_task, replay_buffer)
    return replay_buffer

"""
def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        co = observations[i]['camera_orientation']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        co = np.array(list(co.values()))
        output.append(dict(image=image,camera_orientation=co))
    return output
"""

def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))
    return output

if __name__ == "__main__":
    observation_key = 'image'
    variant = dict(env = 'Widow250DoubleDrawerOpenGraspNeutral-v0')
    import roboverse
    env = roboverse.make(variant['env'], transpose_image=True)
    
    args = lambda:0 #RANDOM Object
    args.buffer = 1
    path = '/nfs/kun1/users/asap7772/cog_data/'
    buffers = []
    ba = lambda x, p=1, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))

    if args.buffer == 0:
        ba('closed_drawer_prior.npy',y='zero')
        ba('drawer_task.npy')
    elif args.buffer == 1:
        ba('closed_drawer_prior.npy',y='zero')
        ba('drawer_task.npy',y='zero')
    elif args.buffer == 2:
        ba('closed_drawer_prior.npy',y='zero')
        ba('drawer_task.npy',y='noise')
    elif args.buffer == 3:
        ba('closed_drawer_prior.npy',y='noise')
        ba('drawer_task.npy',y='zero')
    elif args.buffer == 4:
        ba('closed_drawer_prior.npy',y='noise')
        ba('drawer_task.npy',y='noise')
    variant['buffer'] = buffers
    
    if variant['buffer'] is not None:
        variant['prior_buffer'] = buffers[0]
        variant['task_buffer'] = buffers[1]

    replay_buffer = load_data_from_npy_chaining(variant, env, observation_key, duplicate=True)
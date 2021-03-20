import matplotlib.pyplot as plt
import numpy as np

def plot_traj(imgs, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.set_figheight(5*rows)
    fig.set_figwidth(5*cols)
    for img, ax in zip(imgs, axes.ravel()):
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    return fig

p = '/nfs/kun1/users/asap7772/cog_data/closed_drawer_prior.npy'
save_path = '/home/asap7772/cog/images/closed_drawer_prior'
num_traj = 10

with open(p, 'rb') as f:
    data = np.load(f, allow_pickle=True)

for i in range(num_traj):
    imgs = [x['image'] for x in data[i]['observations']] 
    fig = plot_traj(imgs, 1, len(imgs))
    print(save_path+str(i)+'.png')
    plt.savefig(save_path+str(i)+'.png')

import ipdb; ipdb.set_trace()

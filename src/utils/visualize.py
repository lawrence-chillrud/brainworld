import matplotlib.pyplot as plt
import numpy as np
from .reader import read_scan, find_crop, crop_by

def find_tumorous_slice(data):
    return np.argmax(data.sum(axis=tuple(range(data.ndim - 1))))

def visualize_scan(id, slice=None, crop=True, resize=False, modalities=['flair', 't1', 't1ce', 't2'], plot_scan=True, plot_seg=True, plot_overlay=True, colours=['none', '#a62d60', '#f6d543', '#f1731d'], labels=['Healthy brain tissue', 'Necrotic tumor core', 'Peritumoral edematous/invaded tissue', 'GD-enhancing tumor'], figsize=(8, 10)):
    '''
    Visualizes .nii.gz file located in path.
    
    Examples
    --------
    visualize_scan(2) # visualize patient 2's most tumourous (heuristically) scan (if no slice is passed, most tumorous slice (heuristically) will be displayed)
    visualize_scan(0, 70) # visualize patient 0's slice 70
    '''

    # error handling (ensuring correct inputs provided)
    assert set(modalities).issubset({'flair', 't1', 't1ce', 't2'})
    assert plot_scan or plot_seg or plot_overlay
    assert len(colours) == 4
    assert colours[0] == 'none'

    # set up relevant paths
    if crop:
        sample = read_scan(id, modalities[-1])
        xmin, xmax, ymin, ymax, _, _ = find_crop(sample)

    # set up way to keep track of what should be plotted
    ims = np.array(['scan', 'seg', 'overlay'])
    plots = np.array([plot_scan, plot_seg, plot_overlay])
    ims_to_plot = ims[plots]

    # set up colour maps (seg_cmap should have black background)
    seg_cmap = plt.cm.colors.ListedColormap(['black'] + colours[1:])
    cmaps = np.array(['gray', seg_cmap, 'gray'])[plots]

    if slice is None:
        slice = find_tumorous_slice(read_scan(id, 'seg'))
    
    rows, cols = len(modalities), len(ims_to_plot)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                ax[i, j].set_ylabel(modalities[i].upper())
            
            if i == 0:
                ax[i, j].set_title(ims_to_plot[j])

            ax[i, j].tick_params(axis='both', which='both', length=0, labelsize=0)
            ax[i, j].set_aspect(1)

            if ims_to_plot[j] == 'seg':
                im = read_scan(id, 'seg')
            else:
                im = read_scan(id, modalities[i])
            
            if crop:
                im = crop_by(im, xmin, xmax, ymin, ymax, 0, im.shape[-1])
            
            im_s = im[:, :, slice]
            ax[i, j].imshow(im_s, cmap=cmaps[j])

            if ims_to_plot[j] == 'overlay':
                seg = read_scan(id, 'seg')
                if crop:
                    seg = crop_by(seg, xmin, xmax, ymin, ymax, 0, seg.shape[-1])
                ax[i, j].imshow(seg[:, :, slice], cmap=plt.cm.colors.ListedColormap(colours), alpha=0.3)

    # if legend needed, display it
    if 'seg' or 'overlay' in ims_to_plot:
        fig.legend(bbox_to_anchor=(0.5, 1), handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in colours[1:]], labels=labels[1:], loc='upper center')
    
    fig.suptitle(f"Patient ID: {id}, slice number: {slice}", y = 1.02)
    plt.show()

def plot_patch_grid(patches, figsize=(10, 10), seg=False):
    '''
    Plots grid of patches.
    '''
    rows, cols = len(patches), len(patches[0])
    _, ax = plt.subplots(rows, cols, figsize=figsize)
    cmap = 'gray'
    if seg:
        cmap = plt.cm.colors.ListedColormap(['black', '#a62d60', '#f6d543', '#f1731d'])

    for i in range(rows):
        for j in range(cols):
            ax[i, j].tick_params(axis='both', which='both', length=0, labelsize=0)
            ax[i, j].set_aspect(1)
            ax[i, j].imshow(patches[i][j], cmap=cmap)

    plt.show()

def plot_single_patch(patches, i, j, seg=False):
    '''
    Plots single patch.
    '''
    _, ax = plt.subplots()
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    ax.set_aspect(1)
    cmap = 'gray'
    if seg:
        cmap = plt.cm.colors.ListedColormap(['black', '#a62d60', '#f6d543', '#f1731d'])
    ax.imshow(patches[i][j], cmap=cmap)
    plt.show()

def plot_slice(slice, seg=False):
    '''
    Plots slice.
    '''
    _, ax = plt.subplots()
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    ax.set_aspect(1)
    cmap = 'gray'
    if seg:
        cmap = plt.cm.colors.ListedColormap(['black', '#a62d60', '#f6d543', '#f1731d'])
    ax.imshow(slice, cmap=cmap)
    plt.show()
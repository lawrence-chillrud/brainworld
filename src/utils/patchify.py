import numpy as np

def patchify_slice(slice, patch_size=60, stride=60):
    '''
    Break a given 2D slice into patches of shape patch_size x patch_size.
    Patches are taken with stride stride, so that there is overlap between patches.
    Returns a 2D grid of patches, where patches[i][j] is the patch at row i, column j.
    '''
    patches = []
    for i in range(0, slice.shape[0], stride):
        row = []
        for j in range(0, slice.shape[1], stride):
            patch = slice[i:i+patch_size, j:j+patch_size]
            row.append(patch)
        patches.append(row)
    return patches

def _get_3channel_seg(seg):
    channels = [np.expand_dims(1*(seg == label), axis=-1) for label in [1, 4, 2]]
    return np.squeeze(np.stack(channels, axis=-1))

def _pad_patch(p, patch_size=32):
    current_shape = p.shape
    if len(current_shape) == 4:
        expected_shape = (patch_size, patch_size, patch_size, current_shape[-1])
    else:
        expected_shape = (patch_size, patch_size, patch_size)
    
    padding = [(0, max(0, expected_shape[i] - current_shape[i])) for i in range(len(current_shape))]    
    return np.pad(p, padding)

def patchify_scan(scan, seg=None, patch_size=32, overlap=0.75, thresh=0.1):
    '''
    Break a given scan into 3D patches of shape patch_size x patch_size x patch_size x num_channels,
    with overlap percent overlap between 3D patches. 
    Throw away all patches that don't have at least thresh % of its values (voxels) as non-zero.
    If a seg is supplied then patchify along the tumor, else patchify along the whole brain.
    It is expected that scan is the result of read_mpMRI, 
    while seg is the output of read_scan (4d and 3d numpy ndarrays, respectively).

    Returns
    -------
    scan_patches, seg_patches (lists of patches of equal length) if seg supplied, else just scan_patches
    '''
    # error handling
    assert len(scan.shape) == 4
    if seg is not None: assert len(seg.shape) == 3

    # compute number of patches along each axis
    num_patches_xyz = [int(np.ceil((shape_i - patch_size) / (patch_size*(1 - overlap)))) + 1 for shape_i in scan.shape[:-1]]

    scan_patches = []
    if seg is not None: seg_patches = []

    for i in range(num_patches_xyz[0]):
        for j in range(num_patches_xyz[1]):
            for k in range(num_patches_xyz[2]):
                # compute start and end indices for patch
                start_x = int(i*patch_size*(1 - overlap))
                end_x = min(start_x + patch_size, scan.shape[0])
                start_y = int(j*patch_size*(1 - overlap))
                end_y = min(start_y + patch_size, scan.shape[1])
                start_z = int(k*patch_size*(1 - overlap))
                end_z = min(start_z + patch_size, scan.shape[2])

                # extract patch
                scan_patch = scan[start_x:end_x, start_y:end_y, start_z:end_z, :]
                if scan_patch.shape[:-1] != (patch_size, patch_size, patch_size):
                    scan_patch = _pad_patch(scan_patch, patch_size=patch_size)
                
                if seg is not None: 
                    # patchify according to the tumor
                    seg_patch = seg[start_x:end_x, start_y:end_y, start_z:end_z]
                    if seg_patch.shape != (patch_size, patch_size, patch_size):
                        seg_patch = _pad_patch(seg_patch, patch_size=patch_size)

                    if np.sum(seg_patch != 0) < thresh*patch_size**3:
                        # skip patch if it contains less than thresh% tumor pixels
                        continue
                    # turn seg into 3 channel binary mask
                    seg_patch = _get_3channel_seg(seg_patch)
                    seg_patches.append(seg_patch)
                else:
                    # patchify according to the brain tissue
                    if np.sum(scan_patch[:, :, :, 0] != 0) < thresh*patch_size**3: 
                        # skip patch if it contains less than thresh% informative pixels (i.e., brain tissue)
                        continue
                
                scan_patches.append(scan_patch)
    
    if seg is not None:
        return np.stack(scan_patches, axis=0), np.stack(seg_patches, axis=0)
    else:
        return np.stack(scan_patches, axis=0), None
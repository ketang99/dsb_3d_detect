import numpy as np
import h5py
import os
import math
import pandas as pd
import tifffile
import csv
import json

def print_tree(name, obj):
    """
    Recursive function that analyses the tree structure of the keys in a given ims file (ims files are based on
    hdf5 format).
    :param name: key in parent layer (str)
    :param obj: datastructure that should be analysed
    :return:
    """
    # check if object is an instance of some final classes or if one more layer should be analysed
    if isinstance(obj, (h5py.Dataset, np.ndarray, np.uint64, np.float32)):
        # return information the final layer instance
        return {"type": str(type(obj)), "shape": obj.shape, "dtype": str(obj.dtype)}
    else:
        # check if the current objects holds attribute items
        try:
            # read child items
            children = {**{key: print_tree(key, val) for key, val in obj.attrs.items()}}
        except:
            # initialize an empty dictionary for child items
            children = {}
        # check if the current object holds items
        try:
            # read child items and update the child object dictionary
            children.update({key: print_tree(key, val) for key, val in obj.items()})
        finally:
            # update the child objects and return the final dictionary of child items
            return {**children}

def select_coords(coords, extmin, extmax):

    if coords.shape[0] != 0:

        # print(coords.shape[0])
        x_conform = np.logical_and(coords[:,0] >= extmin[0], coords[:,0] <= extmax[0]).reshape((coords.shape[0],1))
        y_conform = np.logical_and(coords[:,1] >= extmin[1], coords[:,1] <= extmax[1]).reshape((coords.shape[0],1))
        z_conform = np.logical_and(coords[:,2] >= extmin[2], coords[:,2] <= extmax[2]).reshape((coords.shape[0],1))
    
        selected_inds = np.all(np.concatenate((x_conform,y_conform,z_conform), axis=1), axis=1)
        selected_coords = coords[selected_inds]
    # print(selected_coords.shape)
    else:
        selected_coords = np.array([])
        selected_inds = np.array([])
    
    return selected_coords, selected_inds
    

def coords_to_3d(coords_dict, extmin, extmax, rescaled_dims, scaling_factor=4):
    '''
    Parameters
    ----------
    
    coords_dict : np.array()
        A dictionary with the keys 'CoordsXYZR' and 'RadiusYZ'
    extmin : np.array()
        The minimum bounds in um of the cropped region. Indices are X,Y,Z.
    extmax : np.array()
        The max bounds in um of the cropped region. Indices are X,Y,Z.
    rescaled_dims : np.array()
        The size in pixels of the cropped image according to extmax and extmin. Indices are X,Y,Z.
    scaling_factor : factor by which the radius of the spot should be scaled. Default=4.

    Returns
    -------
    Numpy array [len(coords),6] that contains bounding boxes for each spot. Order is Z2Y2X2.

    '''
    
    selected_coords, selected_inds = select_coords(coords_dict['CoordsXYZR'], extmin, extmax)
    # print(selected_coords.shape, len(selected_coords), selected_inds.shape)
        
    if selected_coords.shape[0] != 0:

        bbox = np.zeros((len(selected_coords), 6))
        radii = coords_dict['CoordsXYZR'][selected_inds,-1]
        sgn = -1 * scaling_factor

        for i in range(6):
            idx = math.floor(i/2)
            bbox[:, i] = selected_coords[:,idx] + sgn*radii  
            sgn = -sgn
        
        # if len(coords_dict['RadiusYZ']) == 0:
        #     radii = coords_dict['CoordsXYZR'][selected_inds,-1]
        #     sgn = -1 * scaling_factor
        #     for i in range(6):
        #         idx = math.floor(i/2)
        #         bbox[:, i] = selected_coords[:,idx] + sgn*radii  
        #         sgn = -sgn
            
        # else:   #HSPCs have radii along Y and Z
        #     sgn = -1 * scaling_factor
        #     for i in range(6):
        #         idx = math.floor(i/2)
        #         if i < 2: # X radius
        #             bbox[:, i] = selected_coords[:,idx] + sgn*coords_dict['CoordsXYZR'][selected_inds,-1]
        #         elif i < 4: # Y radius
        #             bbox[:, i] = selected_coords[:,idx] + sgn*coords_dict['RadiusYZ'][selected_inds,0]
        #         else:  # Z radius
        #             bbox[:, i] = selected_coords[:,idx] + sgn*coords_dict['RadiusYZ'][selected_inds,1]
        #         sgn = -sgn
            
        # Force the bboxes to be within the bounds determined by extmin and extmax
        for i in range(3):  # order XYZ
            for j in range(2):  # fix to lower then upper bounds
                if j == 0:
                    bbox[:,2*i+j][bbox[:,2*i+j] < extmin[i]] = extmin[i]
                else:
                    bbox[:,2*i+j][bbox[:,2*i+j] > extmax[i]] = extmax[i]
        
        '''
        Now we have bbox which has the lower and upper bounds of each box in um. Order: X2Y2Z2.
        This needs to be converted to integer voxel values
        '''
        # bbox_um = bbox.copy()
        range_arr = extmax - extmin  # XYZ
        for i in range(3):
            bbox[:, 2*i:2*i+2] = np.round((bbox[:, 2*i:2*i+2] - extmin[i]) / range_arr[i], decimals=3) * rescaled_dims[i]
            bbox.round()
    
        # Change order to Z2Y2X2
        bbox[:,[0,1,-2,-1]] = bbox[:, [-2,-1,0,1]]

    else:
        bbox = np.array([])
    
    return bbox.astype(int)

def get_binary_img(img_shape, bbox):
    
    '''
    Returns a binary image of the same shape as the image (ZYX)
    '''
    out = np.zeros(img_shape).astype(np.uint8)
    
    for i in range(bbox.shape[0]):
        out[bbox[i,0]:bbox[i,1]+1, bbox[i,2]:bbox[i,3]+1, bbox[i,4]:bbox[i,5]+1] = 128
        
    return out.astype(np.uint8)

def get_ims_metadata(filename, r_scaling, desired_cnames = ['DAPI', 'Ctnnal1 GFP', 'HLF tdT']):
    '''
    Parameters
    ----------
    filename : name of the imaris dataset to be loaded
    r_scaling : scaling factor for BB radius
    desired_cnames : the desired channel names in the correct order for combo_img (see next function)

    Returns
    -------
    Metadict: dictionary with metadata about the ims file and channels to be used
    '''
    with h5py.File(filename, 'r') as f:
        
        print('Getting metadata')
        extmin = []
        extmax = []
        for i in range(3):
            valmin = f['DataSetInfo/Image'].attrs[f'ExtMin{i}'].tobytes().decode('ascii', 'decode')
            valmax = f['DataSetInfo/Image'].attrs[f'ExtMax{i}'].tobytes().decode('ascii', 'decode')
            # print(f'ExtMin{i} = {valmin}, ExtMax{i} = {valmax}')
            extmin.append(valmin)
            extmax.append(valmax)
    
        extmin = np.asarray(extmin, dtype='float')
        extmax = np.asarray(extmax, dtype='float')
        
        imcrop_dims = []
        for imdim in ['Z', 'Y', 'X']:
            imd = int(f['DataSetInfo/Image'].attrs[imdim].tobytes().decode('ascii', 'decode'))
            imcrop_dims.append(imd)
    
        resos = (extmax - extmin)[-1::-1] / imcrop_dims
    
        # dict0 = {'CoordsXYZR': f['Scene/Content/Points0/CoordsXYZR'][:], 'RadiusYZ': []}
        # dict1 = {'CoordsXYZR': f['Scene/Content/Points1/CoordsXYZR'][:], 'RadiusYZ': f['Scene/Content/Points1/RadiusYZ'][:]}

        print(desired_cnames, type(desired_cnames))
        channels = np.zeros(len(desired_cnames))
        cnames = []
        # channels = []
            # get channel indices in the order of desired_cnames
        for i in range(6):
            ci = f'Channel {i}'
            if ci in list(f['DataSetInfo'].keys()):
                cname = f[f'DataSetInfo/Channel {i}'].attrs['Name'].tobytes().decode('ascii', 'decode')
                # print(i, cname)
                for j, name in enumerate(desired_cnames):
                    # name1 = name.split(' ')[0]
                    # cname1 = cname.split(' ')[0]
                    # if len(name1) > 5:
                    #     if cname1 == name1[:len(cname)-1].lower():
                    #         channels[j] = i
                    # else:
                    #     if cname1.lower() == name1.lower():
                    #         channels[j] = i
                    if cname.lower() == name[:len(cname)].lower():
                        channels[j] = i
                        cnames.append(cname)
                        # channels.append(i)

        # channels = np.array(channels)

        metadict = {'filename': filename, 'extmin_zyx': extmin[-1::-1], 'extmax_zyx': extmax[-1::-1], 'imdims_zyx': imcrop_dims, 
                        'resolution': resos, 'channel_id': channels.astype(int), 'channel_names': cnames, 'r_scaling': r_scaling}
        
    return metadict

def get_image(filename, r_scaling=4, desired_cnames = ['DAPI', 'Ctnnal1 GFP', 'HLF tdT'], return_img=True):

    metadict = get_ims_metadata(filename, r_scaling, desired_cnames)
    
    with h5py.File(filename, 'r') as f:

        spot_dict = {}
        # No need for below code as RadiusYZ is ~ extmin_yz, not the spot radius
        for p in ['Points0', 'Points1']:
            p_dict = {}
            if not np.isin(p, list(f['Scene/Content'].keys())):
                p_dict['CoordsXYZR'] = np.array([])
                p_dict['RadiusYZ'] = np.array([])
            else:
                if np.isin('CoordsXYZR', list(f[f'Scene/Content/{p}'].keys())):
                    p_dict['CoordsXYZR'] = f[f'Scene/Content/{p}/CoordsXYZR'][:]
                    if np.isin('RadiusYZ', list(f[f'Scene/Content/{p}'].keys())):
                        p_dict['RadiusYZ'] = f[f'Scene/Content/{p}/CoordsXYZR'][:]
                    else:
                        p_dict['RadiusYZ'] = np.array([])    
                else:
                    p_dict['CoordsXYZR'] = np.array([])
                    p_dict['RadiusYZ'] = np.array([])

            spot_dict[p] = p_dict
        
        # dict0 = {'CoordsXYZR': f['Scene/Content/Points0/CoordsXYZR'][:], 'RadiusYZ': []}
        # dict1 = {'CoordsXYZR': f['Scene/Content/Points1/CoordsXYZR'][:], 'RadiusYZ': f['Scene/Content/Points1/RadiusYZ'][:]}

        channels = metadict['channel_id']
        # print(channels)
        extmin = metadict['extmin_zyx'][-1::-1]
        extmax = metadict['extmax_zyx'][-1::-1]
        imcrop_dims = metadict['imdims_zyx']

        if len(channels) == 0:
            print('There must be at least one channel!')
            combo_img = np.array([])
        else:
            if return_img:
                for n,c in enumerate(channels):
                    channel_name = f[f'DataSetInfo/Channel {c}'].attrs['Name'].tobytes().decode('ascii', 'decode')
                    print(f'Reading image for channel {c}: {channel_name}')
                    img = f[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {c}/Data'][:imcrop_dims[0], 
                                                                                       :imcrop_dims[1], 
                                                                                       :imcrop_dims[2]].astype(np.uint8)
                    s = img.shape
                    # print(n,img.shape, np.unique(img)[0:5])
                    if n == 0:
                        combo_img = np.empty((len(channels), s[0],s[1],s[2]), dtype=np.uint8)
                        print('Created combo_img blank array')
                        
                    combo_img[n] = img
                    print(f'  Successfully written channel {c} to combo_img')
                    del img
                    
                print('Image with channels created')
    
            else:
                combo_img = np.array([])

        # print(spot_dict['Points0'].keys())
        # print(np.shape(spot_dict['Points0']['CoordsXYZR']))
        # print(spot_dict['Points1'].keys())
        # input to the below for imcrop_dims needs to be XYZ
        bbox0 = coords_to_3d(spot_dict['Points0'], extmin, extmax, imcrop_dims[-1::-1], scaling_factor=r_scaling)
        bbox1 = coords_to_3d(spot_dict['Points1'], extmin, extmax, imcrop_dims[-1::-1], scaling_factor=r_scaling)
        print('bboxes found')
        
        print('Complete')

    return combo_img, bbox0, bbox1, metadict

def get_max_bbox_sizes(bbox):

    diff_mat = np.zeros((len(bbox),3))
    for i in range(3):
        ind = 2*i
        diff_mat[:,i] = bbox[:,ind+1] - bbox[:,ind]

    out = np.max(diff_mat, axis=0)

    return out

def find_patch_starts(single_step_size, single_imdim, single_sp):

    starts = [0]
    p_current = starts[0]
    p_current+=single_step_size
    thresh = single_imdim - single_sp

    while thresh >= p_current:
        starts.append(p_current)
        p_current+=single_step_size

    starts = np.asarray(starts)
    
    return starts.astype(int)

def find_overlap_and_patch_type(current_patch, s_p, max_bb, step_sizes, z_starts, y_starts, x_starts):

    poss_steps = np.zeros((3,2))
    poss_steps[:,0] = -step_sizes
    poss_steps[:,1] = step_sizes

    valid_bools = np.empty((3,2)).astype(bool)
    valids = 0
    for i in range(3):
        for j in range(2):
            new_patch = current_patch
            new_patch[i] += poss_steps[i,j]

            if np.any(np.isin(z_starts, new_patch[0])) and np.any(np.isin(y_starts, new_patch[1])) and np.any(np.isin(x_starts, new_patch[2])):
                valids += 1
                valid_bools[i,j] = True
            else:
                valid_bools[i,j] = False

    patch_vol = np.prod(s_p)
    print(valids)
    
    if valids == 3:  # corner case
        patch_type = 'corner'
        overlap_vol = patch_vol - np.prod(s_p - max_bb)
        
    elif valids == 4:  # edge case
        patch_type = 'edge'
        test_bool = np.all(valid_bools, axis=1)
        if test_bool[0]:  # z axis has two available directions
            overlap_vol = patch_vol - (s_p[0]-2*max_bb[0])*(s_p[1]-max_bb[1])*(s_p[2]-max_bb[2])
            #overlap_vol = patch_vol - np.prod(np.asarray([[s_p - max_bb], [s_p - 2*max_bb]], [s_p]).reshape((3,2)), axis=1)
        elif test_bool[1]:
            overlap_vol = patch_vol - (s_p[0]-max_bb[0])*(s_p[1]-2*max_bb[1])*(s_p[2]-max_bb[2])
        elif test_bool[2]:
            overlap_vol = patch_vol - (s_p[0]-max_bb[0])*(s_p[1]-max_bb[1])*(s_p[2]-2*max_bb[2])

    elif valids == 5:
        patch_type = 'face'
        test_bool = np.all(valid_bools, axis=1)
        if not test_bool[0]:
            overlap_vol = patch_vol - (s_p[0]-max_bb[0])*(s_p[1]-2*max_bb[1])*(s_p[2]-2*max_bb[2])
        elif not test_bool[1]:
            overlap_vol = patch_vol - (s_p[0]-2*max_bb[0])*(s_p[1]-max_bb[1])*(s_p[2]-2*max_bb[2])
        elif not test_bool[2]:
            overlap_vol = patch_vol - (s_p[0]-2*max_bb[0])*(s_p[1]-2*max_bb[1])*(s_p[2]-max_bb[2])

    elif valids == 6:
        patch_type = 'surrounded'
        overlap_vol = patch_vol - np.prod(s_p - 2*max_bb)

    return overlap_vol.astype(int), patch_type

def patch_bbox(current_patch, s_p, bbox):

    if len(bbox) != 0:
        bb_shifted = np.copy(bbox)
        # print(bb_shifted.shape)
        # print(len(bb_shifted))
        bb_shifted[:,[0,1]] -= current_patch[0]
        bb_shifted[:,[2,3]] -= current_patch[1]
        bb_shifted[:,[4,5]] -= current_patch[2]
        # print('BB shifted shape: ',bb_shifted.shape)
        # print(bb_shifted[:5,:])
    
        valid_bbs = bb_shifted[np.all(bb_shifted>=0, axis=1)] 
        valid_bbs_low = valid_bbs[:,[0,2,4]]
        valid_bbs_up = valid_bbs[:,[1,3,5]]
        # print(valid_bbs)
        # print(valid_bbs_low.shape, valid_bbs_up.shape)
        
        z_inside = np.logical_and(valid_bbs_low[:,0]<s_p[0], valid_bbs_up[:,0]<=s_p[0])
        y_inside = np.logical_and(valid_bbs_low[:,1]<s_p[1], valid_bbs_up[:,1]<=s_p[1])
        x_inside = np.logical_and(valid_bbs_low[:,2]<s_p[2], valid_bbs_up[:,2]<=s_p[2])
    
        # print(z_inside.shape, y_inside.shape, x_inside.shape)
        # bb_inside = np.logical_any(bbox_lower < current_patch_upper, bbox_upper <= current_patch_upper)
        bb_inside = np.all(np.array([[z_inside],[y_inside],[x_inside]]), axis=0)
        # patch_bb_low = bb_lower[bb_inside,:]
        # patch_bb_up = bb_upper[bb_inside,:]
        # print('bb inside shape:', bb_inside.shape, len(bb_inside))
        # print(bb_inside[0])
        patch_bb = valid_bbs[bb_inside.reshape((bb_inside.shape[1],))]

    else:
        patch_bb = np.array([])

    return patch_bb

def generate_patches(img, datafile, img_name, filedir, bbox0, bbox1, s_p, max_bb, step_sizes, z_starts, y_starts, x_starts):

    '''
    datafile: already created. An h5 file that stores the patches and their data
                this file has groups Patches and Metadata
                Patches stores individual patches
                Each individual patch has the patch crop and info
                The info contains the bounding boxes

    This function will write to an h5 file. It'll create all possible patches and store their respective
    data including bboxes. 
    '''
    # make paths for patches and ground truth (for each cell type)
    # patch_dir = f'DSB/training_images/'
    # if not os.path.isdir(patch_dir):
    #     os.makedirs(f'{patch_dir}/patches')

    # if not os.path.isdir(f'{patch_dir}/patch_data'):
    #     os.makedirs(f'{patch_dir}/patch_data')

    # bbox_dir = f'DSB/training_images/ground_truth/{bb_type}'
    # if not os.path.isdir(bbox_dir):
    #     os.makedirs(bbox_dir)

    if not os.path.isfile(f'{filedir}/{datafile}'):
        with h5py.File(f'{filedir}/{datafile}', 'w') as f:
            f.create_group('Patches')
            # pg.create_group('Train')
            # pg.create_group('Test')
            mg = f.create_group('Metadata')
            # mg.create_group()

    with h5py.File(f'{filedir}/{datafile}', 'a') as f:
        start_num_patch = len(list(f['Patches'].keys())) + 1
        num_patches = len(list(f['Patches'].keys()))
        print(f' Start number patches: {start_num_patch}')
        print(img.shape)
        has_cell = []
        for z in z_starts:
            if z == z_starts[0]:
                print('Entered for loop z')
            z_id = str(z).zfill(5)
            for y in y_starts:
                if y == y_starts[0]:
                    print('  Entered for loop y')
                y_id = str(y).zfill(5)
                for x in x_starts:
                    if x == x_starts[0]:
                        print('    Entered for loop x')
                        
                    num_patches+=1
                    
                    x_id = str(x).zfill(5)
                    patch_id = f'{z_id}_{y_id}_{x_id}'
                    patch_dict = {}
                    patch_dict['Filename'] = img_name
                    patch_dict['Patch_ID'] = patch_id
                    current_patch = np.array([z,y,x])
                    patch_dict['Start_pos'] = current_patch
    
                    # implement function to find patch type and overlapping area
                    overlap, p_type = find_overlap_and_patch_type(current_patch, s_p, max_bb, step_sizes, z_starts, y_starts, x_starts)
                    patch_dict['Overlap'] = overlap
                    patch_dict['Patch_type'] = p_type

                    # Write and save csv for patch's bboxes
                    if len(bbox0) == 0 and len(bbox1) == 0:
                        patch_bboxes = np.array([[]])
                        patch_bbtypes = np.array([])
                        patch_dict['bbox'] = patch_bboxes
                        patch_dict['Cell_type'] = patch_bbtypes
                    
                    else:
                        # has_cell.append(num_patches)
                        all_bboxes = [bbox0,bbox1]
                        print(len(all_bboxes))
                        patch_bbtypes = []
                        patch_bboxes = np.empty((0,6))
                        num_cells = 0
                        for i, bb in enumerate(all_bboxes):
                            # if len(bb) == 0:
                            #     break
                            curr_bb = patch_bbox(current_patch, s_p, bb)
                            if len(curr_bb) != 0:
                                # has_cell.append(num_patches)
                                patch_bboxes = np.concatenate((patch_bboxes, curr_bb), axis=0)
                                for j in range(curr_bb.shape[0]):
                                    patch_bbtypes.append(i)
                                    num_cells+=1

                        patch_bbtypes = np.asarray(patch_bbtypes)

                        patch_dict['bbox'] = patch_bboxes.astype(int)
                        patch_dict['Cell_type'] = patch_bbtypes.astype(int)
    
                    
                    patch_dict['Num_cells'] = num_cells

                    pg = f['Patches'].create_group(f'Patch {num_patches}')
                    pi = pg.create_group('Patch info')
                    for key, value in patch_dict.items():
                        if isinstance(value, str):
                            pi.create_dataset(key, data=np.array([value]).astype('S'))
                        else:
                            pi.create_dataset(key, data=value)

                    
                    # check if patch needs to be padded with 0s
                    # rewrite this with some boolean array cuz imo there's some logical error here
                    patchbounds = np.zeros((3,2))
                    patchbounds[:,0] = current_patch
                    patchbounds[:,1] = current_patch + s_p
                    patchbounds = patchbounds.astype(int)
                    img_shape = img.shape[1:]

                    patch_edge = patchbounds[:,1] > img_shape

                    for i in range(3):
                        if patch_edge[i]:
                            patchbounds[i,1] = img_shape[i]

                    patch = np.zeros((img.shape[0], s_p[0], s_p[1], s_p[2])).astype(np.uint8)
                    crop = img[:, patchbounds[0,0]:patchbounds[0,1], patchbounds[1,0]:patchbounds[1,1], 
                                patchbounds[2,0]:patchbounds[2,1]]
                    patch[:, :crop.shape[1], :crop.shape[2], :crop.shape[3]] = crop

                    pg.create_dataset('Patch', data=patch.astype(np.uint8))
                    
                    del patch, crop

        patch_num_range = [start_num_patch, num_patches]

    return patch_num_range, has_cell
                

def generate_training_dataset(imaris_dir, nn_home_dir, channels = [2,4], px_r = 25):

    '''
    Parameters:
    ----------
    imaris_dir : full directory where all the imaris datasets are stored
    nn_home_dir : full directory that contains the training set and ground truths
    channels : a list containing the channel ID numbers that we want to save for training 
    px_r : the desired half-width of the bounding box along X and Y

    Returns:
    '''

    if not os.isdir(nn_home_dir + '/training_images'):
        os.mkdir(nn_home_dir + '/training_images')

    if not os.isdir(nn_home_dir + '/gt'):
        os.mkdir(nn_home_dir + '/gt')
    
    for imaris_file in os.listdir(imaris_dir):
        if imaris_file[-4:] == '.ims':
            channel_string = 'channels_'
            for i in len(channels):
                if i != len(channels) - 1:
                    channel_string += f'{channels[i]}_'
                else:
                    channel_string += f'{channels[i]}'
            dataset_name = imaris_file[0:-4] + f'_px_lateral_{px_r}' + channel_string
            
            combo_img, bbox = registration_multi_channel(imaris_dir + f'/{imaris_file}', channels, px_r)

            tifffile.imsave(f'{nn_home_dir}/training_images/{dataset_name}.tif', combo_img)
            df.to_csv(f'{nn_home_dir}/gt/{dataset_name}.csv', index=False)
            
    

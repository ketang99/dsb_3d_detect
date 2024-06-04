import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
import time
import collections
import random
import json
import reg_functions as reg
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from split_combine import SplitComb

class HLFBoneMarrowCells(Dataset):

    def __init__(self, datafile, root_dir, config, split_comber, phase='Train', transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datafile = datafile
        self.root_dir = root_dir
        self.fileloc = os.path.join(self.root_dir, self.datafile)
        self.phase = phase
        self.label_mapping = LabelMappingAll(config, phase=self.phase)
        self.split_comber = split_comber
        self.transform = transform

        #with h5py.File(self.fileloc, 'r') as f:
        #    self.pids = list(f[f'Patches/{self.phase}'].keys())

        # self.pids = self.pids[:24]
        # np.random.shuffle(self.pids)
        # self.pids = self.pids[:2]
        if self.phase == 'Train':
            self.pids = ['Patch 199', 'Patch 199']
        elif self.phase == 'Val':
            self.pids = ['Patch 580', 'Patch 580']

    def __len__(self):

        return len(self.pids)

    def __getitem__(self, idx):
        '''
        nzhw: right now it gives the patch dimensions. 
              Have it contain the number of bboxes n and their respective z,y,x lengths in vx
        bboxes: return these in the form [zc, yz, xc] where c means center

        Returns:
        patch
        bboxes : first three cols are centers for ZYX. Last column is the size along X
        bbox_size : size along Z,Y,X
        celltypes : 
        '''
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with h5py.File(self.fileloc, 'r') as f:
            
            patchdict = {}
            # pids = list(f[f'Patches/{self.phase}'].keys())
            # print(pids[idx])
            # pg = f[f'Patches/{self.phase}/{self.pids[idx]}']
            pg = f[f'Patches/{self.phase}/{self.pids[idx]}']

            # print(pg['Patch'][()].shape)
            pi = pg['Patch info']
        
            patch = pg['Patch'][()]
            # patch = patch[:,:192,:192,:192]
            # bbtypes = pi['Cell_type'][()]
            bbox = pi['bbox'][()]
            ploc = pi['Start_pos'][()]
            filename = pi['Filename'][0].tobytes().decode('ascii', 'decode')

        # patch, nzhw = self.split_comber.split(patch)
        bboxes = np.zeros((bbox.shape[0], 4))
        # bbox_size = np.zeros((bbox.shape[0], 3))
        for i in range(3):
            bboxes[:,i] = 0.5 * (bbox[:,2*i] + bbox[:, 2*i+1])
            # for now let's do just one dimension's size: make it X
            
        bboxes[:,-1] = bbox[:,-1] - bbox[:,-2]
        
        print('patch id: ', self.pids[idx])
        # print('bboxes: ', bboxes)
        # labels = []
        # for i in range(len(bboxes)):
        #     target = bboxes[i]
        #     label = self.label_mapping(patch.shape[-3:], target, bboxes)
        #     labels.append(label)
        #     del label

        pos,label = self.label_mapping(patch.shape[-3:], bboxes)

        if self.transform:
            sample = self.transform(sample)

        print('Patch shape: ', patch.shape)
        # print('before norm min and max of patch: ', np.min(patch), np.max(patch))
        patch = (patch-128.0)/128.0
        # print('min and max of patch: ', np.min(patch), np.max(patch))

        return {'patch': torch.from_numpy(patch).to(torch.float), 
                'bboxes': bboxes,
                'label': torch.from_numpy(label),
                'start_pos': ploc, 
                'patch_idx': idx}

def custom_collate_dict(batch):
    collated_batch = {}
    for key in batch[0].keys():
        # print(key)
        if key == 'bboxes' or key == 'start_pos' or key == 'patch_idx':
            # Handle variable number of bounding boxes
            collated_batch[key] = [sample[key] for sample in batch]
        elif key == 'label' or key =='patch':
            collated_batch[key] = torch.stack([sample[key] for sample in batch])
    
    return collated_batch


def custom_collate_dict_new(batch):
    collated_batch = {}
    batch_size = len(batch)  # Get the actual batch size
    # print('loader collate batch size: ', batch_size)
    # Handle keys with variable-length lists
    variable_length_keys = ['bboxes', 'start_pos', 'patch_idx']
    for key in variable_length_keys:
        collated_batch[key] = [sample[key] for sample in batch]

    # Handle keys with fixed-size tensors (e.g., 'label' and 'patch')
    fixed_size_keys = ['label', 'patch']
    for key in fixed_size_keys:
        collated_batch[key] = torch.stack([sample[key] for sample in batch])

    # Ensure that the batch size is consistent across all keys
    assert all(len(collated_batch[key]) == batch_size for key in collated_batch.keys()), "Inconsistent batch size"

    return collated_batch


class LabelMappingAll(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'Train':
            self.th_pos = config['th_pos_train']
        elif phase == 'Val':
            self.th_pos = config['th_pos_val']

            
    def __call__(self, input_size, bboxes):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        struct = generate_binary_structure(3,1)      
        
        output_size = []
        for i in range(3):
            assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)

        output_size = [int(x) for x in output_size]
        
        label = np.zeros(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 1
                label[:,:,:, i, 0] = binary_dilation(label[:,:,:, i, 0].astype('bool'),structure=struct,iterations=1).astype('float32')
                                                      
        
        label = label-1

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        # up till this point, the specific bounding box does not matter, after this it will
        # print('bboxes: ', bboxes)
        if len(bboxes) == 0:
            poss = []
        else:
            poss = []
            for i in range(len(bboxes)):
                target = bboxes[i]
                pos = self.find_pos(target, bboxes, oz, oh, ow)
                poss.append(pos)
            
                dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
                dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
                dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
                dd = np.log(target[3] / anchors[pos[3]])
                label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
                # print(i, pos, [1, dz, dh, dw, dd])
            
        return poss,label

    
    def find_pos(self, target, bboxes, oz, oh, ow):
        
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        struct = generate_binary_structure(3,1)

        offset = ((stride.astype('float')) - 1) / 2
        
        if np.isnan(target[0]) or len(target) == 0:
            return []
            
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True 
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]

        return pos

def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]
        
        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)
        
        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0
        
        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        #if th > 0.4:
         #   if np.sum(mask) == 0:
          #      print(['iou not large', iou.max()])
           # else:
            #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw

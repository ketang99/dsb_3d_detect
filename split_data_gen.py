import numpy as np
import h5py
import os
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
import reg_functions as reg
import random
import json

class SplitData():
    def __init__(self, filenames, s_p, homedir, filedir, h5_name, num_desired=[0,0,0,0], r_scaling=4, single_red=True, find_max_bb=False, max_bb = np.array([23,67,87])):
        self.filenames = filenames
        self.s_p = s_p
        self.homedir = homedir
        self.filedir = filedir
        self.h5name = h5_name
        self.find_max_bb = find_max_bb
        self.fixed_max_bb = max_bb
        self.single_red = single_red
        self.r_scaling = 4
        self.patch_id = []
        self.image_id = []
        self.patch_w_cell = []
        self.num_desired = num_desired
            

        if single_red:
            self.desired_cnames = ['HLF tdT']
        else:
            self.desired_cnames = ['DAPI', 'Ctnnal1 GFP', 'HLF tdT']

    def forward(self):

        print('Get metadata, bboxes and patch starts for all ims files')
        # get metadata, bboxes and patch specs and starting points
        self.get_meta_bbox()
        self.get_max_bb(self.bbox1, self.find_max_bb)
        patch_starts = self.get_zyx_patch_starts()  # all images

        if self.num_desired[0] != 0:
           for i in range(len(self.filenames)):
               # patch_starts[i] = patch_starts[i][np.random.randint(0,len(patch_starts[i]),self.num_desired[i])] 
               patch_starts[i] = patch_starts[i][np.random.choice(len(patch_starts[i]),self.num_desired[i])]

        print('Getting cells present')
        cells_present = []
        for i in range(len(self.filenames)):
            cells_present.append(self.has_cell(patch_starts[i], i))

        print('Getting indices for train/test/val')
        allcases = [[],[],[]]
        for i in range(len(self.filenames)):
            alls = self.get_patch_indices(cells_present[i])
            for j in range(3):
                allcases[j].append(np.concatenate((alls[j][0], alls[j][1])))
                # print(len(allcases[j][i]))

        print('Saving patches and phase IDs')
        # select patches based on the index for each imaris file
        phase_id = [[],[],[]]
        for i, file in enumerate(self.filenames):
            print('')
            print(file)
            print('')
            # instead of reading the ims file for each phase, read the image once
            img,_,_,_ = reg.get_image(f'{self.homedir}/{self.filedir}/{self.filenames[i]}', self.r_scaling, self.desired_cnames, return_img=True)                
            for p, phase in enumerate(['Train','Test','Val']):
                print(f'Saving patches for {phase} phase')
                save_patches = allcases[p][i]
                print('Combo shape: ', img.shape)
                pids = self.save_split_patches(img, i, patch_starts[i], save_patches, cells_present[i], phase)
                phase_id[p].append(np.array(pids))

                print('Files saved for current phase')
        
        for p, phase in enumerate(['Train', 'Test', 'Val']):
            with h5py.File(f'{self.homedir}/{self.filedir}/{self.h5name}', 'a') as f:
                tg = f['Metadata']
                tg.create_dataset(f'{phase} ID', data=np.concatenate(phase_id[p]))

            print('')

        print('Saving overall metadata')
        with h5py.File(f'{self.homedir}/{self.filedir}/{self.h5name}', 'a') as f:
            mg = f['Metadata']
            mg.create_dataset('Patch ID', data=self.patch_id)
            mg.create_dataset('Image ID', data=self.image_id)
            mg.create_dataset('Patch dimensions', data=self.s_p)
            mg.create_dataset('Step size', data=self.step_size)
            mg.create_dataset('Max BB', data=self.max_bb)
       
            for ims_i in range(len(self.filenames)):
                if f'Image {ims_i}' not in list(f['Metadata'].keys()):
                    print('not there')
                    mg = f['Metadata'].create_group(f'Image {ims_i}')
                else:
                    print('already there')
                    mg = f[f'Metadata/Image {ims_i}']
                print(mg.keys())
                print(self.metadata[ims_i].keys())
                self.dict_to_h5(mg, self.metadata[ims_i])

        print('Making json file')
        with h5py.File(f'{self.homedir}/{self.filedir}/{self.h5name}', 'r') as f:
        # with h5py.File('12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims', 'r') as f:
            # get the tree structure of the file
            tree = reg.print_tree(self.h5name, f)
        
            # write the tree to a JSON file
            with open(f'{self.homedir}/{self.filedir}/JSON_{self.h5name}.json', 'w') as file:
                json.dump(tree, file, indent=4)

            # print status message
            print(f'Finished IMS file tree analysis!')

        print('End')
        

    def save_split_patches(self, img, ims_i, patch_starts, idcs, has_cell, phase):

        '''
        Metadata: patch ID, startpos, type
        Each image: patch IDs corresponding to this image
        '''
        idc_patch_starts = patch_starts[idcs]
        idc_cells = has_cell[idcs]
        
        if not os.path.isfile(f'{self.filedir}/{self.h5name}'):
            with h5py.File(f'{self.homedir}/{self.filedir}/{self.h5name}', 'w') as f:
                pg = f.create_group('Patches')
                pg.create_group('Train')
                pg.create_group('Test')
                pg.create_group('Val')
                f.create_group('Metadata')

        with h5py.File(f'{self.homedir}/{self.filedir}/{self.h5name}', 'a') as f:

            # if f'Image {ims_i}' not in list(f['Metadata'].keys()):
            #     print('not there')
            #     mg = f['Metadata'].create_group(f'Image {ims_i}')
            # else:
            #     print('already there')
            #     mg = f[f'Metadata/Image {ims_i}']
            # print(mg.keys())
            # print(self.metadata[ims_i].keys())
            # self.dict_to_h5(mg, self.metadata[ims_i])

            # print(f['Patches'].keys())
            fg = f['Patches']
            tg = fg[phase]
            num_init = len(list(fg['Train'].keys())) + len(list(fg['Test'].keys())) + len(list(fg['Val'].keys()))
            num_patches = 0
            
            for i in range(len(idc_patch_starts)):
                num_patches+=1
                current_patch = idc_patch_starts[i]
                self.patch_w_cell.append(idc_cells[i])
                self.patch_id.append(num_patches + num_init)
                self.image_id.append(ims_i)

                cg = tg.create_group(f'Patch {num_patches + num_init}')

                patch_dict = {}
                patch_dict['Filename'] = self.filenames[ims_i]
                patch_dict['Patch_ID'] = num_patches + num_init
                patch_dict['Start_pos'] = current_patch

                # Write and save csv for patch's bboxes
                if self.single_red:
                    bbox0 = self.bbox0
                else:
                    bbox0 = self.bbox0[ims_i]
                
                if len(bbox0) == 0 and len(self.bbox1[ims_i]) == 0:
                    patch_bboxes = np.array([[]])
                    patch_bbtypes = np.array([])
                    patch_dict['bbox'] = patch_bboxes
                    patch_dict['Cell_type'] = patch_bbtypes
                
                else:
                    # has_cell.append(num_patches)
                    all_bboxes = [bbox0, self.bbox1[ims_i]]
                    # print(len(all_bboxes))
                    patch_bbtypes = []
                    patch_bboxes = np.empty((0,6))
                    num_cells = 0
                    for i, bb in enumerate(all_bboxes):
                        # if len(bb) == 0:
                        #     break
                        curr_bb = reg.patch_bbox(current_patch, self.s_p, bb)
                        if len(curr_bb) != 0:
                            # has_cell.append(num_patches)
                            patch_bboxes = np.concatenate((patch_bboxes, curr_bb), axis=0)
                            for j in range(curr_bb.shape[0]):
                                patch_bbtypes.append(i+1)
                                num_cells+=1

                patch_bbtypes = np.asarray(patch_bbtypes)

                patch_dict['bbox'] = patch_bboxes.astype(int)
                patch_dict['Cell_type'] = patch_bbtypes.astype(int)
                patch_dict['Num_cells'] = num_cells

                pi = cg.create_group('Patch info')
                self.dict_to_h5(pi, patch_dict)

                patchbounds = np.zeros((3,2))
                patchbounds[:,0] = current_patch
                patchbounds[:,1] = current_patch + self.s_p
                patchbounds = patchbounds.astype(int)
                img_shape = img.shape[1:]

                patch_edge = patchbounds[:,1] > img_shape

                for i in range(3):
                    if patch_edge[i]:
                        patchbounds[i,1] = img_shape[i]

                patch = np.zeros((img.shape[0], self.s_p[0], self.s_p[1], self.s_p[2])).astype(np.uint8)
                crop = img[:, patchbounds[0,0]:patchbounds[0,1], patchbounds[1,0]:patchbounds[1,1], 
                            patchbounds[2,0]:patchbounds[2,1]]
                patch[:, :crop.shape[1], :crop.shape[2], :crop.shape[3]] = crop

                cg.create_dataset('Patch', data=patch.astype(np.uint8))
                del patch, crop

        return [num_init, num_patches + num_init]


    def dict_to_h5(self, pg, save_dict):
        for key, value in save_dict.items():
            # print(key)
            if isinstance(value, str):
                pg.create_dataset(key, data=np.array([value]).astype('S'))
            else:
                pg.create_dataset(key, data=value)
        
    def get_max_bb(self, bbox1, find_max_bb):
        if find_max_bb:
            all_max_bb = []
            for i in range(len(self.filenames)):
                # all_max_bb.append(reg.get_max_bbox_sizes(bbox0[i]))
                all_max_bb.append(reg.get_max_bbox_sizes(bbox1[i]))
    
            self.max_bb = np.max(np.array(all_max_bb), axis=0)
        else:
            self.max_bb = self.fixed_max_bb

        self.step_size = self.s_p - self.max_bb

    def get_meta_bbox(self):
        metadata = []
        bbox0 = []
        bbox1 = []
        for file in self.filenames:
            # print(file)
            meta = reg.get_ims_metadata(f'{self.homedir}/{self.filedir}/{file}', r_scaling=self.r_scaling, desired_cnames = ['HLF tdT'])
            metadata.append(meta)
            extmin = meta['extmin_zyx'][-1::-1]
            extmax = meta['extmax_zyx'][-1::-1]
            imcrop_dims = meta['imdims_zyx']
            # with h5py.File(f'{homedir}/{filedir}/{file}', 'r') as f:
            with h5py.File(f'{self.homedir}/{self.filedir}/{file}', 'r') as f:
                p0 = {}
                p0['CoordsXYZR'] = f[f'Scene/Content/Points0/CoordsXYZR'][:]
                p1 = {}
                p1['CoordsXYZR'] = f[f'Scene/Content/Points1/CoordsXYZR'][:]
        
                bbox0.append(reg.coords_to_3d(p0, extmin, extmax, imcrop_dims[-1::-1], scaling_factor=4))
                bbox1.append(reg.coords_to_3d(p1, extmin, extmax, imcrop_dims[-1::-1], scaling_factor=4))

        self.metadata = metadata
        if self.single_red:
            self.bbox0 = np.array([])
        else:
            self.bbox0 = bbox0
        self.bbox1 = bbox1

    def get_zyx_patch_starts(self):
    # patch starts for all images
        patch_starts = []
        s_p = self.s_p
        for i in range(len(self.filenames)):
            st = {}
            imdims = self.metadata[i]['imdims_zyx']
            zs = reg.find_patch_starts(self.step_size[0], imdims[0], s_p[0])
            ys = reg.find_patch_starts(self.step_size[1], imdims[1], s_p[1])
            xs = reg.find_patch_starts(self.step_size[2], imdims[2], s_p[2])
        
            starts_i = []
            for z in zs:
                for y in ys:
                    for x in xs:
                        starts_i.append([z,y,x])
        
            starts_i = np.array(starts_i)
            patch_starts.append(starts_i)

        return patch_starts

    # write a function that finds whether or not a patch has a cell/bbox
    def has_cell(self, one_patch_starts, iter):
        # all patches for one image
    
        has_cell = []
        if len(self.bbox0) == 0:
            all_bboxes = [self.bbox0,self.bbox1[iter]]
        else:
            all_bboxes = [self.bbox0[iter],self.bbox1[iter]]
        # print(len(all_bboxes[0]),len(all_bboxes[1]))
        
        for i in range(len(one_patch_starts)):
            current_patch = one_patch_starts[i]
            
            if len(all_bboxes[0]) == 0 and len(all_bboxes[1]) == 0:
                print('both bboxes blank')
                patch_bboxes = np.array([[]])
                patch_bbtypes = np.array([])
                # patch_dict['bbox'] = patch_bboxes
                # patch_dict['Cell_type'] = patch_bbtypes
            
            else:
                # has_cell.append(num_patches)
                # print(len(all_bboxes))
                patch_bbtypes = []
                patch_bboxes = np.empty((0,6))
                num_cells = 0
                c_already = False
                zero_check1 = False
                for i, bb in enumerate(all_bboxes):
                    # if len(bb) == 0:
                    #     break
                    curr_bb = reg.patch_bbox(current_patch, self.s_p, bb)
                    if len(curr_bb) != 0:
                        if not c_already:
                            has_cell.append(1)
                            c_already = True
                        patch_bboxes = np.concatenate((patch_bboxes, curr_bb), axis=0)
                        for j in range(curr_bb.shape[0]):
                            patch_bbtypes.append(i)
                            num_cells+=1
                    else:
                        if zero_check1:
                            has_cell.append(0)
                        else:
                            zero_check1 = True
    
        return np.array(has_cell)
    
    
    def patch_split(self, idcs, train_pct=0.8, test_pct=0.1, val_pct=0.1):
        splits = [int(train_pct * len(idcs)), int(test_pct * len(idcs)), int(val_pct * len(idcs))]
        train_cell = idcs[:splits[0]]
        test_cell = idcs[splits[0]:splits[0] + splits[1]]
        val_cell = idcs[splits[0]+splits[1]:splits[0]+splits[1]+splits[2]]
    
        return train_cell, test_cell, val_cell
    
    def get_patch_indices(self, cells_present):
        num_p = len(cells_present)
        num_cell = np.count_nonzero(cells_present)
        num_nc = round(num_cell * 0.25*1/0.75)
        print(num_cell, num_nc)
        cell_idx = np.where(np.array(cells_present)!=0)
        # print(cell_idx)
        no_cell_idx = np.where(np.array(cells_present)==0)
        # print(no_cell_idx, len(no_cell_idx[0]), len(cell_idx))
        cell_idx = cell_idx[0]
        no_cell_idx = no_cell_idx[0]
        print(f'num of no cells: {len(no_cell_idx)}, actual no cells = {num_nc}')
        rc = np.random.choice(len(no_cell_idx), num_nc, replace=False).astype(int)
        no_cell_idx = no_cell_idx[rc]
        # print(type(no_cell_idx), type(cell_idx))
    
        # print(cell_idx)
        
        np.random.shuffle(cell_idx)
        # now divide the patches with cells among training, testing and validation (80,10,10)
        traincell, testcell, valcell = self.patch_split(cell_idx)
    
        # do the same for cells without patches
        np.random.shuffle(no_cell_idx)
        trainno, testno, valno = self.patch_split(no_cell_idx)
    
        return [traincell,trainno], [testcell,testno], [valcell,valno]

ND_HLF.py 
- performs training and validation of the network in order to detect HSPCs in main()
- saves the loss metrics as numpy arrays at the end of each training epoch
- contains training and validation functions; these are called in main()

data_red_dsb.py 
- contains the dataset class for the dataloader as well as functions to generate the ground truth 5D array used for training
- normalizes the 3D image patches (intensity range [0,255] in uint8 form) to values ranging from [-1,1]
- converts the bounding boxes to the form [z_center, y_center, x_center, width]
- currently hardcoded to use a single patch for training (twice, with a batch size of 2) and another single patch for validation

reg_functions.py 
- contains functions that help retrieve 3D images, bounding boxes and image metadata, as well as patch generation

split_data_gen.py 
- contains class that generates and saves patches of size [128,128,128] which will be used for training

red_dsb/layers.py
- contains the loss class which is used during training and validation in ND_HLF.py
- cls loss uses nn.BCELoss() and reg loss uses 
- contains the PostRes3D class which forms the residual blocks used in the network

red_dsb/training/classifier/net_detector3.py
- contains the network's layers

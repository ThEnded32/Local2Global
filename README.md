# Local2Global
Official repository of  the paper "Local2Global: UNet with Hierarchical Attention Mechanisms for Improved MR Image Inpainting" 

This code base is built upon the official repository of the BraTS 2025 Inpainting Challenge: https://github.com/BraTS-inpainting/2025_challenge
Please follow their instructions before starting to use this repository.
After setting the challenge repository, download the "train_Local2GlobalV3.py", "my_codes/local2globalUNet.py", and "my_codes/my_blocks.py" files under the "baseline" directory.

These three files are the main files. Also, the high-resolution training code is in the main branch named "train_Local2GlobalV3_highres.py". ## NOTE: There is a padding operation on line 60 ##
You can access the checkpoint submitted to the challenge from: https://drive.google.com/file/d/1KcfTtE692yTsok-HtGKOxwTK8RkSQNq8/view?usp=sharing

The other files, including the atlas, augmentations, etc., will be put in the "other_files" directory with its own README file, which stores the explanations. 

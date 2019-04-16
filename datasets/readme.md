# how to create the multi-view dataset with .tiff files from original dataset with .dcm files

## Steps:
1. Find function 'create_multiview_dataset_multi_preocess' from path './datasets/multiview.py'.
    There are four parameters for this function: 'method', 'data_dir', 'des_dir' and 'num_workers'
    - keep 'method' and 'num_workers' unmodified in 'multiview.py'
    - 'data_dir' denotes from where to read data, namely the path or directory saving original dcm files
        - Eg. data_dir = "/data/ugui0/antonio-t/CPR_20181206"
    - 'des_dir' denotes to where to save data, namely the path of directory saving the .tiff files
        - Eg. des_dir = "/data/ugui0/antonio-t/CPR_20181206_tmp"
 
2. make sure skimage is installed on your system
3. run 'python datasets/multiview.py' which contains the main function under the root folder PlaqueDetection_20181127. 
then tiff files can be generated in the des_dir
    - example main function in multiview.py file
    ```    
        data_dir = "/data/ugui0/antonio-t/CPR_20181206"
        des_dir = "/data/ugui0/antonio-t/CPR_20181206_t"
        create_multiview_dataset_multi_preocess(create_multiview_dataset, data_dir, des_dir, num_workers=1)
    ```
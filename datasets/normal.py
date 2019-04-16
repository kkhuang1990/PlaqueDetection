import os
import os.path as osp
from os import listdir
import numpy as np
import pydicom as dicom
from skimage import io
from multiprocessing import Pool
from utils import dcm2hu, hu2gray, rgb2gray, rgb2mask, centra_crop, mask2gray


def resave_multi_preocess(method, data_dir, des_dir, num_workers=24):
    """ resave data into desired format for faster read
    Args:
        method: function, use which method to resave the data
        data_dir: string, from where to read data
        des_dir: string, to where to save data
        num_workers: int, how many processes in parallel
    """
    args = []

    for sample in listdir(data_dir):
        print("Processing {}".format(sample))
        # for sample in listdir(data_dir):
        sample_path = osp.join(data_dir, sample)
        if any([sample.startswith(prefix)] for prefix in ['IRB', 'S']) and osp.isdir(sample_path):
            series = [s for s in listdir(sample_path) if not s.startswith('.')]
            series = sorted(series, key=lambda x: int(x[1:]))
            series_path = osp.join(sample_path, series[0])
            exclusions = ['.tiff', 'conf']
            phases = [phase for phase in sorted(listdir(series_path))
                      if not any([phase.endswith(ex) for ex in exclusions]) and not phase.startswith('.')]

            for phase in phases:
                phase_path = osp.join(series_path, phase)
                des_phase_path = osp.join(des_dir, sample, phase)
                args.append((phase_path, des_phase_path))

    pool = Pool(processes=num_workers)
    print("{} CPUs are used".format(num_workers))
    pool.starmap(method, args)
    pool.close()


#########################################################################################################
# resave dcm to tiff without data augmentation
def dcm2tiff_per_artery_wo_augment(data_dir, des_dir):
    """ resave dcm file into tiff image for each artery
    Args:
        data_dir: string, from where to read data
        des_dir: string, to where to write data into
    """

    print("processing {}".format(data_dir.split('/')[-3:]))
    phase = data_dir.split('/')[-1]
    # load masks
    mask_dir = data_dir + 'conf'
    mask_files = [file for file in listdir(mask_dir) if file.startswith('I0') and file.endswith('.tiff')]
    mask_files = sorted(mask_files, key=lambda x: int(x.split('.')[0][2:]))

    # load images
    img_dir = data_dir
    img_files = [file for file in listdir(img_dir) if file.startswith('I0')]
    img_files = sorted(img_files, key=lambda x: int(x.split('.')[0][2:]))

    if len(mask_files) > 0 and len(img_files) > 0:
        slice_info_file = osp.join(mask_dir, 'sliceinfo.txt')
        start, end, risks, sig_stenosises, pos_remodelings, napkin_rings = get_slice_info(slice_info_file)

        mask_files, img_files = mask_files[start:end], img_files[start:end]

        img = np.stack([dcm2hu(dicom.read_file(osp.join(img_dir, file))) if file.endswith('.dcm')
                        else io.imread(osp.join(img_dir, file))
                        for file in img_files])

        mask = np.stack([rgb2mask(io.imread(osp.join(mask_dir, file))) for file in mask_files])

        des_image_path = osp.join(des_dir, 'image')
        des_mask_path = osp.join(des_dir, 'mask')

        if not osp.exists(des_image_path):
            os.makedirs(des_image_path)
        if not osp.exists(des_mask_path):
            os.makedirs(des_mask_path)

        for slice, label in zip(img, mask):
            io.imsave(osp.join(des_image_path, "{:03d}.tiff".format(start + 1)), slice)
            io.imsave(osp.join(des_mask_path, "{:03d}.tiff".format(start + 1)), label)
            start += 1

        # save annotation information
        np.savetxt(osp.join(des_dir, "risk_labels.txt"), risks, fmt='%d')
        np.savetxt(osp.join(des_dir, "significant_stenosis_labels.txt"), sig_stenosises, fmt='%d')
        np.savetxt(osp.join(des_dir, "positive_remodeling_labels.txt"), pos_remodelings, fmt='%d')
        np.savetxt(osp.join(des_dir, "napkin_ring_labels.txt"), napkin_rings, fmt='%d')

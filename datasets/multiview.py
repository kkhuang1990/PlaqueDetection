# _*_ coding: utf-8 _*_

""" Functions for creating dataset of multi-view slices """

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from skimage import io
import pydicom as dicom
import os
import os.path as osp
from os import listdir
import numpy as np
from utils import dcm2hu
from multiprocessing import Pool
from skimage import transform
from operation import get_slice_info
from utils import rgb2gray

def create_multiview_dataset(data_dir, des_dir):
    """ create multi-view dataset with  abscissa, ordinate and applicate (x, y and z) respectively
        Main steps are listed as below:
        (1) read slice information from sliceinfo.txt, which includes effective range, risk, positive remodeling,
            napkin ring sign, significant stenosis and et al
        (2) extract angle_0 and angle_90 CPR image for both input and annotation and resize into proper size
        (3) extract patches along each CPR image and resave them into corresponding directories
    :param src_dir: str, from where to read artery slices
    :param des_dir: str, to where to write multi-view slices into
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

        # make sure at least one sample is effective
        if start < end:
            # read dcm information from I01.dcm and angle_0.tiff file
            dcm = dicom.read_file(osp.join(data_dir, 'I01.dcm'))
            s_shape = (dcm.Rows, dcm.Columns)
            s_thick = dcm.SliceThickness
            print("slice thickness: {}".format(s_thick))
            pix_space = [float(ele) for ele in dcm.PixelSpacing]

            s_spcae = [space * size for space, size in zip(pix_space, s_shape)]
            print("slice shape: {}, pixel space: {}".format(s_shape, pix_space))
            num_slices = s_spcae[0] / s_thick
            CPR_angle0_file = osp.join(data_dir + 'conf', 'angle_0.tiff')
            CPR_angle0 = io.imread(CPR_angle0_file).astype(np.uint8)
            num_rows = len(CPR_angle0)
            print("# of rows: {}, # of slices: {}".format(num_rows, len(img_files)))
            rows_per_slice = num_rows / len(img_files)
            num_rows_patch = int(num_rows * num_slices / len(img_files))
            print("{} rows are necessary to extract a cube".format(num_rows_patch))
            num_rows_patch = int(round(num_rows_patch / 32)) * 32

            # resave slices along the applicate axis
            image = np.stack([dcm2hu(dicom.read_file(osp.join(img_dir, file))) if file.endswith('.dcm')
                            else io.imread(osp.join(img_dir, file))
                            for file in img_files])

            mask = np.stack([rgb2gray(io.imread(osp.join(mask_dir, file))) for file in mask_files])

            info_range = []
            non_calcified = np.zeros_like(risks)
            calcified = np.zeros_like(risks)
            for axis_name in ['applicate', 'abscissa', 'ordinate']:
                for data_name in ['image', 'mask']:
                    des_path = osp.join(des_dir, axis_name, data_name)
                    if not osp.exists(des_path):
                        os.makedirs(des_path)
                    data = image if data_name == 'image' else mask
                    if axis_name != 'applicate':
                        if axis_name == 'abscissa':
                            cpr = data[:, s_shape[0]//2, :]
                        elif axis_name == 'ordinate':
                            cpr= data[:, :, s_shape[0]//2]

                        if data_name == 'image':
                            cpr_rescale = transform.resize(cpr, (num_rows, s_shape[0]), mode='reflect',
                                                           preserve_range=True)
                            cpr_rescale = cpr_rescale.astype(np.int16)
                            io.imsave(osp.join(des_dir, axis_name, 'cpr_image.tiff'), cpr_rescale)
                        else:
                            cpr_rescale = transform.resize(cpr, (num_rows, s_shape[0]), mode='reflect',
                                                           preserve_range=True, order=0)
                            cpr_rescale = cpr_rescale.astype(np.uint8)
                            io.imsave(osp.join(des_dir, axis_name, 'cpr_mask.tiff'), cpr_rescale)

                    for s_inx in range(start, end):
                        c_inx = 2 + int(rows_per_slice * s_inx)
                        lower, upper = c_inx - num_rows_patch // 2, c_inx + num_rows_patch // 2
                        # only save samples with complete multiple view slices
                        if lower >= 0 and upper <= num_rows:
                            if axis_name == "applicate":
                                io.imsave(osp.join(des_path, "{:03d}.tiff".format(s_inx + 1)), data[s_inx])
                                # save slice info along applicate axis
                                if data_name == 'mask':
                                    info_range.append(s_inx - start)
                                    if np.sum(data[s_inx] == 76) != 0:
                                        non_calcified[s_inx - start] = 1
                                    if np.sum(data[s_inx] == 151) != 0:
                                        calcified[s_inx - start] = 1
                            else:
                                batch = cpr_rescale[lower:upper]
                                io.imsave(osp.join(des_path, "{:03d}.tiff".format(s_inx + 1)), batch)

            # save annotation information
            np.savetxt(osp.join(des_dir, "risk_labels.txt"), risks[info_range], fmt='%d')
            np.savetxt(osp.join(des_dir, "significant_stenosis_labels.txt"), sig_stenosises[info_range], fmt='%d')
            np.savetxt(osp.join(des_dir, "positive_remodeling_labels.txt"), pos_remodelings[info_range], fmt='%d')
            np.savetxt(osp.join(des_dir, "napkin_ring_labels.txt"), napkin_rings[info_range], fmt='%d')
            np.savetxt(osp.join(des_dir, "non_calcified_plaque_labels.txt"), non_calcified[info_range], fmt='%d')
            np.savetxt(osp.join(des_dir, "calcified_plaque_labels.txt"), calcified[info_range], fmt='%d')

def remove_redundant_slice_applicate(data_dir):
    """ remove redundant slices along applicate axis
        so that the applicate and other axes can have the same number of slices
    """
    # print("processing {}".format(data_dir.split('/')[-3:]))

    noncals = np.genfromtxt(osp.join(data_dir, "non_calcified_plaque_labels.txt")).astype(np.uint8)

    # for axis_name in ['applicate', 'abscissa']:
    for data_name in ['image', 'mask']:
        des_path = osp.join(data_dir, 'applicate', data_name)
        if not osp.exists(des_path):
            print("Be careful not to remove necessary files/folders")
            break

        src_path = des_path.replace('applicate', 'abscissa')
        if not osp.exists(src_path):
            break

        slice_files = [file for file in listdir(des_path) if file.endswith('.tiff') and not file.startswith('.')]
        slice_files = sorted(slice_files, key=lambda x: int(x.split('.')[0]))

        for file in slice_files:
            src_file_path = osp.join(src_path, file)
            des_file_path = osp.join(des_path, file)
            if not osp.exists(src_file_path):
                print("there still exists redundant file, please remove it!!!")
                os.remove(des_file_path)

    slice_files = [file for file in listdir(des_path) if file.endswith('.tiff') and not file.startswith('.')]
    assert len(slice_files) == len(noncals), "number of slices should be the same as non-calcification records"

def remove_redundant_slice_applicate_multi_preocess(method, data_dir, num_workers=1):
    """ resave data into desired format for faster read
        WARNING!!! please do not use multiple process to read data
    Args:
        method: function, use which method to resave the data
        data_dir: string, from where to read data
        des_dir: string, to where to save data
        num_workers: int, how many processes in parallel
    """
    args = []
    # samples = ['IRB00000000001269600000000211194620110823HOSPNO']
    # for sample in samples:
    for sample in listdir(data_dir):
        print("Processing {}".format(sample))
        # for sample in listdir(data_dir):
        sample_path = osp.join(data_dir, sample)
        if osp.isdir(sample_path) and (sample.startswith('S') or sample.startswith('IRB')):
            exclusions = ['.tiff', 'conf']
            phases = [phase for phase in sorted(listdir(sample_path))
                      if not any([phase.endswith(ex) for ex in exclusions]) and not phase.startswith('.')]

            for phase in phases:
                phase_path = osp.join(sample_path, phase)
                args.append(phase_path)

    pool = Pool(processes=num_workers)
    print("{} CPUs are used".format(num_workers))
    pool.map(method, args)
    pool.close()


def create_multiview_dataset_multi_preocess(method, data_dir, des_dir, num_workers=1):
    """ resave data into desired format for faster read
        WARNING!!! please do not use multiple process to read data
    Args:
        method: function, use which method to resave the data
        data_dir: string, from where to read data
        des_dir: string, to where to save data
        num_workers: int, how many processes in parallel
    """
    args = []

    samples = ['S2189281c5_S1ff973cbc08376_20181120']
    for sample in samples:
    # for sample in listdir(data_dir):
        print("Processing {}".format(sample))
        sample_path = osp.join(data_dir, sample)
        if osp.isdir(sample_path) and (sample.startswith('S') or sample.startswith('IRB')):
            series = [s for s in listdir(sample_path) if not s.startswith('.')]
            series = sorted(series, key=lambda x: int(x[1:]))
            for ser in series:
                series_path = osp.join(sample_path, ser)
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


if __name__ == "__main__":
    # cal_slice_size()
    # CPR_extraction()
    # data_dir = "/data/ugui0/antonio-t/CPR_all"
    # data_dir = "/data/ugui0/antonio-t/01"
    # data_dir = "/data/ugui0/antonio-t/CPR_20181003"
    data_dir = "/data/ugui0/antonio-t/CPR_20190108"
    # data_dir = "/data/ugui0/antonio-t/CPR_20180601/20180601"
    # data_dir = "/data/ugui0/antonio-t/CPR_20180713"
    des_dir = "/data/ugui0/antonio-t/CPR_20190108_misannotation_tmp"
    # des_dir = "/data/ugui0/antonio-t/CPR_multiview"
    create_multiview_dataset_multi_preocess(create_multiview_dataset, data_dir, des_dir, num_workers=1)
    # read_window_info_multi_preocess(read_window_info, data_dir, num_workers=24)

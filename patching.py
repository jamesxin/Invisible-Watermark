import array
import sys
import os
import glob
import math
import re
import argparse
import copy
import csv
from YUV import YUVFile
from vqm import VQM
import logging
import ntpath


__author__ = 'James F'
__version__ = '0.0.1'

############################
# logging
############################

# create logger with 'spam_application'
logger = logging.getLogger('Patch_YUV')
logger.setLevel(logging.DEBUG)
# logger.propagate = False

# create file handler which logs even debug messages
fh = logging.FileHandler('debug.log', mode='w')
#fh.setLevel(logging.DEBUG)
fh.setLevel(logging.WARNING)


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


class Patching:
    """compute PNSR for frames.
    """

    def __init__(self):
        pass

    # patching the watermark
    @staticmethod
    def patching_frame(yuv_wt, yuv_org, blocksize):

        # validation, the size of yuv should dividable by blocksize**2
        if yuv_wt.file_size != yuv_org.file_size:
            print "the data size is not the same"
            return

        if yuv_wt.file_size % (blocksize ** 2) != 0 or yuv_org.file_size % (blocksize ** 2) != 0:
            print "data size: frame1 {0},  frame2 {1} are not divisible by blocksize {2} ".format(yuv_wt.file_size,
                                                                                                  yuv_org.file_size,
                                                                                                  blocksize)
            return

        # create the output mse array
        if yuv_org.bytes_per_sample == 1:
            mse_array = array.array('B')
        elif yuv_org.bytes_per_sample == 2:
            mse_array = array.array('H')

        w = yuv_org.YUV_width
        h = yuv_org.YUV_height
        # only take the y channel
        mse_array = yuv_org.YUV_data[:w * h]
        logger.debug("the original mse_array size is {0}".format(len(mse_array)))
        # left to right and up to bottom, generate a y' with mse

        w_blocks = w / blocksize
        h_blocks = h / blocksize

        scaledown_blocks = 0
        for w_j in xrange(0, w_blocks):
            for h_i in xrange(0, h_blocks):
                block_wt = []
                block_org = []
                for k in range(0, blocksize):
                    startaddr = (h_i * blocksize + k) * w + w_j * blocksize
                    block_wt.extend(yuv_wt.YUV_data[startaddr: (startaddr + blocksize)])
                    block_org.extend(yuv_org.YUV_data[startaddr: (startaddr + blocksize)])

                logger.debug("org is ")
                for i in xrange(0, blocksize*blocksize, blocksize):
                    logger.debug(block_org[i:i+blocksize])
                logger.debug("watermarked is")
                for i in xrange(0,blocksize*blocksize, blocksize):
                    logger.debug(block_wt[i:i+blocksize])

                #################################
                # update watermark based on MAX_AD
                ################################
                L_Diff = 0
                H_Diff = 0
                block_sum = sum(block_org)
                block_average = block_sum*1.0 / len(block_org)
                average_diversity = sum([abs(i - block_average) for i in block_org]) / len(block_org)

                for (pixel_org, pixel_changed) in zip(block_wt, block_org):
                    Diff = pixel_changed - pixel_org
                    L_Diff = Diff if Diff < L_Diff else L_Diff
                    H_Diff = Diff if Diff > H_Diff else H_Diff

                AD = (H_Diff - L_Diff)/2
                # max AD , 10, AD/max_AD to scale down the changes
                # show we only scale down the change on a pixel or entire block?
                # compute the avg of block. if it is quite even, we should use a lower MAX_AD
                # Otherwise we should use a higher MAX_AD
                if block_average < 600:
                    if block_average < 3:
                        MAX_L_H_AD = math.exp(2.28*math.log(block_average)/3.09)
                    else:
                        MAX_L_H_AD = average_diversity * 1.16
                else:
                    MAX_L_H_AD = math.exp(1.1*math.log(block_average)/1.5)
                # should be adjusted by average diversity
                # Max_L_H_AD should be different for the extremely low or high luma
                # The constant value is from experiments (Digital video Quality)
                # for now let us assume the range is from 3-660
                # for the watermarked y < 3, the incr or decr should be e^(2.28*logL/3.09)*L
                # for the watermarked 660 > y > 3, the incr or decr should be 1.16*L
                # for the watermarked y > 660, the incr or decr could be e^(1.1*logL/1.5)*L


                logger.debug(" block {0} {1}".format(w_j, h_i))
                logger.debug("[AD, MAX_L_H_AD] is [{0},{1}]".format(AD, MAX_L_H_AD))

                if AD > MAX_L_H_AD:

                    scaledown_blocks += 1
                    # we need scale down the watermark
                    # assign the mse to the array
                    for h_k in range(0, blocksize):
                        startaddr = (h_i * blocksize + h_k) * w + w_j * blocksize
                        for block_k in range(0, blocksize):
                            pixel = block_org[h_k * blocksize + block_k]
                            mse_array[startaddr + block_k] = int(
                                pixel + (block_wt[h_k * blocksize + block_k] - pixel) * (MAX_L_H_AD * 1.0 / AD))
                        logger.debug(mse_array[startaddr: startaddr + blocksize])

                else:
                    for h_k in range(0, blocksize):
                        startaddr = (h_i * blocksize + h_k) * w + w_j * blocksize
                        block_row = h_k * blocksize
                        for block_k in range(0, blocksize):
                            mse_array[startaddr + block_k] = block_org[block_row + block_k]

        # create a mse array and return it
        logger.debug("the size of mse_array is {0}".format(len(mse_array)))
        logger.debug("{0} blocks ({1}%) are scaledowned.".format(scaledown_blocks, scaledown_blocks * 100 / (w_blocks * h_blocks)))
        logger.debug("#### end of patching")
        return mse_array


def main(argv):

    parser = argparse.ArgumentParser(description="""
    PNSR tool for YUV files.
    """)

    # set the program version
    parser.add_argument('-version', action='version', version='%(prog)s {0}'.format(__version__))

    # specify the input parameters

    # required args
    parser.add_argument(
        '-format',
        action='store',
        dest='yuv_format',
        default='',
        required=True,
        help=' The YUV format.'
    )

    parser.add_argument(
        '-width',
        action='store',
        dest='width',
        type=int,
        default='',
        required=True,
        help='Width of YUV files.'
    )

    parser.add_argument(
        '-height',
        action='store',
        type=int,
        dest='height',
        default='',
        required=True,
        help='Height of YUV files.'
    )

    parser.add_argument(
        '-wm_file',
        action='store',
        dest='wm_file',
        default='',
        required=True,
        help=' watermarked YUV.'
    )


    parser.add_argument(
        '-ref_file',
        action='store',
        dest='ref_file',
        default='',
        required=True,
        help=' The original reference YUV.'
    )

    parser.add_argument(
        '-mse_blocksize',
        action='store',
        dest='mse_blocksize',
        type=int,
        default=0,
        required=True,
        help='''
        When this option is specified  a new yuv file will be generated. It''s y
        channel will be the mse value for each block.'''
    )
    # parse the arguments for the current command line.
    opt = parser.parse_args()

    # validate the input
    if opt.ref_file == '' and opt.wm_file == '':
        print "empty input. Please at least specify either -input_folder or -yuvfiles."
        return 1
    elif os.path.exists(opt.ref_file) and os.path.exists(opt.wm_file):
        print "{0} and {1} are found.".format(opt.ref_file, opt.wm_file)

    else:
        print "input is invalid."
        return 1

    # Setup YUV reader

    ref_YUV = YUVFile(opt.ref_file, opt.yuv_format, opt.width, opt.height)
    wm_YUV = YUVFile(opt.wm_file, opt.yuv_format, opt.width, opt.height)

    if opt.mse_blocksize == 0:
        print "blocksize has to be larger than 0."
        return 1
    else:

        VQM_func = VQM()

        if ref_YUV.bytes_per_sample == 1:
            mse_y_array = array.array('B')
        elif ref_YUV.bytes_per_sample == 2:
            mse_y_array = array.array('H')

        # mse for unpatched frames
        print "ref yuv is {0},  wm_yuv is {1}".format(ref_YUV, wm_YUV)
        mse_y_array = VQM_func.frame_diff_by_block(ref_YUV, wm_YUV, opt.mse_blocksize)


        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(ref_YUV.YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))

            with open(os.path.splitext(ref_YUV.YUV_file)[0]+"_mse.yuv", 'wb') as mse_file:
                # mse_y_array.tofile(mse_file)
                mse_file.write(mse_y_array)

        #########################
        # start patching
        #########################

        #print "the watermarked file (YUVs_list[0].YUV_data:{0}".format(YUVs_list[0].YUV_file)
        #print "the orginal file YUVs_list[1].YUV_data: {0}".format(YUVs_list[1].YUV_file)
        mse_y_array = Patching.patching_frame(wm_YUV, ref_YUV, opt.mse_blocksize)
        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(ref_YUV.YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))
            with open(os.path.splitext(wm_YUV.YUV_file)[0]+"_patched_frame.yuv", 'wb') as mse_file:
                mse_file.write(mse_y_array)

            # we generate a mse_pathed.yuv for testing purpose
            patched_yuv = YUVFile(os.path.splitext(wm_YUV.YUV_file)[0]+"_patched_frame.yuv", opt.yuv_format, opt.width, opt.height)
            patched_mse_y_array = VQM_func.frame_diff_by_block(wm_YUV, patched_yuv, opt.mse_blocksize)

            patched_mse_y_array.extend(wm_YUV.YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))
            with open(os.path.splitext(wm_YUV.YUV_file)[0]+"patched_mse.yuv", 'wb') as mse_file:
                mse_file.write(patched_mse_y_array)


if __name__ == '__main__':
    main(sys.argv)

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
# CRITICAL	50
# ERROR	40
# WARNING	30
# INFO	20
# DEBUG	10
# NOTSET	0
# create logger with 'spam_application'
logger = logging.getLogger('Patch_YUV')
logger.setLevel(logging.DEBUG)
# logger.propagate = False

# create file handler which logs even debug messages
fh = logging.FileHandler('debug.log', mode='w')
fh.setLevel(logging.DEBUG)
#fh.setLevel(logging.WARNING)


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

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

        logger.info("Patching................")
        # validation, the size of yuv should dividable by blocksize**2
        if yuv_wt.file_size != yuv_org.file_size:
            logger.error("the data size is not the same")
            return

        if yuv_wt.file_size % (blocksize ** 2) != 0 or yuv_org.file_size % (blocksize ** 2) != 0:
            logger.error("data size: frame1 {0},  frame2 {1} are not divisible by blocksize {2} ".format(yuv_wt.file_size,
                                                                                                  yuv_org.file_size,
                                                                                                  blocksize))
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
        scaleup_blocks = 0
        normal_blocks = 0

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
                # luma range
                # for 10 bit? it should adapt to 8 bit
                luma_min = 1
                luma_max = 2047

                L_Diff = 0
                H_Diff = 0
                MAX_L_H_AD = 0
                pixel_luma_min = luma_max
                pixel_luma_max = luma_min

                block_sum = sum(block_org)
                block_average = block_sum*1.0 / len(block_org)
                average_diversity = sum([abs(pixel_i - block_average) for pixel_i in block_org]) / len(block_org)

                # since we use average_diversity

                for (pixel_changed, pixel_org) in zip(block_wt, block_org):
                    Diff = pixel_changed - pixel_org
                    L_Diff = Diff if Diff < L_Diff else L_Diff
                    H_Diff = Diff if Diff > H_Diff else H_Diff
                    pixel_luma_min = pixel_org if pixel_org < pixel_luma_min else pixel_luma_min
                    pixel_luma_max = pixel_org if pixel_org > pixel_luma_max else pixel_luma_max

                AD = (H_Diff - L_Diff)*1.0/2
                # AD/max_AD to scale down the changes
                # show we only scale down the change on a pixel or entire block?
                # compute the avg of block. if it is quite even, we should use a lower MAX_AD
                # Otherwise we should use a higher MAX_AD
                # the constant value is based on experiment now


                luma_low_boundary = 60
                luma_high_boundary = 600

                average_diversity = 2 if average_diversity < 2 else average_diversity

                # exception: for some block, even they have a high diversity however, the data are sparse distributed
                # for example, in a range of [0, 100], data are all in the 0, or 100. Such block is not good for hiding data


                if block_average < luma_high_boundary:
                    if block_average < luma_low_boundary:
                         # MAX_L_H_AD = 0.1*math.exp(2.28 * math.log(block_average)/3.09)
                         MAX_L_H_AD = average_diversity * 1.15
                    else:
                        MAX_L_H_AD = average_diversity * 0.25
                else:
                     #MAX_L_H_AD = 0.1*math.exp(1.1*math.log(block_average)/1.5)
                    MAX_L_H_AD = average_diversity * 1.15
                # should be adjusted by average diversity
                # Max_L_H_AD should be different for the extremely low or high luma
                # The constant value is from experiments (Digital video Quality)
                # for now we get the range from experiment. It is from 3-660
                # slight change it to 500

                # for the watermarked y < 3, the incr or decr should be e^(2.28*logL/3.09)
                # for the watermarked 660 > y > 3, the incr or decr should be 0.01*L
                # for the watermarked y > 660, the incr or decr could be e^(1.1*logL/1.5)

                # To Do: how to adjust it by frequency ?
                # We`` can consider

                # MAX_L_H_D should not cause the luma < luma_min or over luma_max

                logger.debug(" block {0} {1}".format(w_j, h_i))
                logger.debug("[AD between WM and ref, MAX_L_H_AD, block_average, average_diversity] is [{0},{1}, {2}, {3}]".format(AD, MAX_L_H_AD, block_average, average_diversity))

                MAX_L_H_AD_1 = (pixel_luma_min -luma_min) if (pixel_luma_min - MAX_L_H_AD) < luma_min else MAX_L_H_AD
                MAX_L_H_AD_2 = (luma_max -pixel_luma_max) if (pixel_luma_max + MAX_L_H_AD) > luma_max else MAX_L_H_AD

                MAX_L_H_AD = MAX_L_H_AD_1 if MAX_L_H_AD_1 < MAX_L_H_AD_2 else MAX_L_H_AD_2



                if AD > MAX_L_H_AD:
                    scaledown_blocks += 1
                elif AD == MAX_L_H_AD:
                    normal_blocks += 1
                else:
                    scaleup_blocks += 1
                #if AD is 0. The watermarked frame did not change anything

                factor = 1
                if AD > MAX_L_H_AD:
                    factor = MAX_L_H_AD * 1.0 / AD
                elif AD == 0:
                    factor = 0
                else:
                    factor = 1.01
                    # we need scale down the watermark
                    # assign the mse to the array

                logger.debug("normalized MAX_L_H_AD is {0} and factor is {1}".format(MAX_L_H_AD, factor))


                for h_k in range(0, blocksize):
                    startaddr = (h_i * blocksize + h_k) * w + w_j * blocksize
                    for block_k in range(0, blocksize):
                        pixel = block_org[h_k * blocksize + block_k]
                        mse_array[startaddr + block_k] = int(
                            pixel + (block_wt[h_k * blocksize + block_k] - pixel) * factor)
                    logger.debug(mse_array[startaddr: startaddr + blocksize])

#                else:
#                    # we should keep the wm block
#                    for h_k in range(0, blocksize):
#                        startaddr = (h_i * blocksize + h_k) * w + w_j * blocksize
#                        block_row = h_k * blocksize
#                        for block_k in range(0, blocksize):
#                            mse_array[startaddr + block_k] = block_wt[block_row + block_k]
        # create a mse array and return it
        #logger.debug("the size of mse_array is {0}".format(len(mse_array)))
        logger.info("{0} blocks ({1}%) are scaledowned.".format(scaledown_blocks, scaledown_blocks * 100 / (w_blocks * h_blocks)))
        logger.info("{0} blocks ({1}%) are scaleuped.".format(scaleup_blocks, scaleup_blocks * 100 / (w_blocks * h_blocks)))
        logger.info("{0} blocks ({1}%) are kept.".format(normal_blocks, normal_blocks * 100 / (w_blocks * h_blocks)))
        logger.info("#### end of patching")
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
        logger.error("empty input. Please at least specify either -input_folder or -yuvfiles.")
        return 1
    elif os.path.exists(opt.ref_file) and os.path.exists(opt.wm_file):
        logger.info("{0} and {1} are found.".format(opt.ref_file, opt.wm_file))

    else:
        logger.error("input is invalid.")
        return 1

    # Setup YUV reader

    ref_YUV = YUVFile(opt.ref_file, opt.yuv_format, opt.width, opt.height, logger)
    wm_YUV = YUVFile(opt.wm_file, opt.yuv_format, opt.width, opt.height, logger)

    if opt.mse_blocksize == 0:
        logger.error("blocksize has to be larger than 0.")
        return 1
    else:

        VQM_func = VQM(logger)

        if ref_YUV.bytes_per_sample == 1:
            mse_y_array = array.array('B')
        elif ref_YUV.bytes_per_sample == 2:
            mse_y_array = array.array('H')

        # mse for unpatched frames
        logger.info("ref yuv is {0},  wm_yuv is {1}".format(ref_YUV.YUV_file, wm_YUV.YUV_file))
        logger.info("#### MSE between ref_yuv and wm_yuv.")
        mse_y_array = VQM_func.frame_diff_by_block(ref_YUV, wm_YUV, opt.mse_blocksize)

        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(ref_YUV.YUV_data[opt.width * opt.height:])
            # print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
            #                                                                                     len(mse_y_array))

            with open(os.path.splitext(ref_YUV.YUV_file)[0]+"_mse.yuv", 'wb') as mse_file:
                # mse_y_array.tofile(mse_file)
                mse_file.write(mse_y_array)

        #########################
        # start patching
        #########################
        logger.info("#### Start to Patch the watermarked frame.")
        # print "the watermarked file (YUVs_list[0].YUV_data:{0}".format(YUVs_list[0].YUV_file)
        # print "the orginal file YUVs_list[1].YUV_data: {0}".format(YUVs_list[1].YUV_file)
        mse_y_array = Patching.patching_frame(wm_YUV, ref_YUV, opt.mse_blocksize)
        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(ref_YUV.YUV_data[opt.width * opt.height:])
            # print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
            # len(mse_y_array))
            with open(os.path.splitext(wm_YUV.YUV_file)[0]+"_patched_frame.yuv", 'wb') as mse_file:
                mse_file.write(mse_y_array)

            # we generate a mse_pathed.yuv for testing purpose
            patched_yuv = YUVFile(os.path.splitext(wm_YUV.YUV_file)[0]+"_patched_frame.yuv", opt.yuv_format, opt.width, opt.height, logger)

            logger.info("#### MSE between ref_yuv and patched_wm_yuv.")
            patched_mse_y_array = VQM_func.frame_diff_by_block(ref_YUV, patched_yuv, opt.mse_blocksize)
            patched_mse_y_array.extend(wm_YUV.YUV_data[opt.width * opt.height:])
            # print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
            # len(mse_y_array))
            with open(os.path.splitext(wm_YUV.YUV_file)[0]+"_patched_mse.yuv", 'wb') as mse_file:
                mse_file.write(patched_mse_y_array)


if __name__ == '__main__':
    main(sys.argv)

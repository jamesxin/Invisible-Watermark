import array
import sys
import os
import glob
import math
import argparse
import csv
from YUV import YUVFile


__author__ = 'James F'
__version__ = '0.0.1'


class VQM:
    """compute visual quality metrix for frames.
        currently support SAD, MSE, PNSR.

    """

    def __init__(self):
        pass

    def mean(self, seq):
        if len(seq) == 0:
            return 0.0
        else:
            return sum(seq) / float(len(seq))

    # summary of absolute difference
    # give a new YUV composed by the differences
    def SAD(self, seq):
        pass

    #
    def sum_square_err(self, data1, data2):
        if len(data1) != len(data2):
            print "Data in different size !"
            sys.exit(0)
        else:
            return sum((a - b) * (a - b) for a, b in zip(data1[:], data2[:]))


    def MSE(self, data1, data2):
        return self.sum_square_err(data1, data2) / len(data1)


    def PSNR(self, mse):
        log10 = math.log10
        if mse == 0:
            mse = sys.float_info.min
        return 10.0 * log10(float(256 * 256) / float(mse))

    # apply the quality metrix on two yuv files
    # it takes two YUV files as input
    def frame_evaluation(self, yuv1, yuv2):
        begin_pos = 0
        end_pos = len(yuv1.YUV_data)
        luma_size = yuv1.luma_size
        chroma_size = yuv1.chroma_size
        frame_size = yuv1.frame_size

        # print "data size: %d, w*h=%d" % (end_pos - begin_pos, luma_size)
        print "evaluating mse.."

        # start of Y U V
        y = begin_pos
        u = y + luma_size
        v = u + chroma_size

        begin = [y, u, v, y]
        end = [u, v, end_pos, end_pos]
        size = [luma_size, chroma_size, chroma_size, frame_size]

        colorspace_mse = [ self.MSE(yuv1.YUV_data[begin[i]:end[i]], yuv2.YUV_data[begin[i]:end[i]]) for i in
                          range(4)]

        colorspace_psnr = [self.PNSR(m) for m in colorspace_mse]
        return colorspace_mse, colorspace_psnr, colorspace_psnr[-1]

    # apply reference quality metrix on a set of yuv files
    def frames_evaluation(self, frames):
        frames_cache = frames.copy()

        for frame_i in frames_cache:
            frames_cache.discard(frame_i)
            for frame_j in frames_cache:
                #print frame_i.YUV_data
                colorplane_mse, colorplane_psnr, frame_psnr = self.frame_evaluation(frame_i, frame_j)


    # output to different files for each metrix, mse, psnr, psnr_y,psnr_u, psnr_v
    def output_csv(self, pnsr_list):
        return

    # compute the mse for two same size data
    def frame_diff_by_block(self, yuv1, yuv2, block_size):

        # validation, the size of frame_data should dividable by blocksize**2
        if yuv1.file_size != yuv2.file_size:
            print "the data size is not the same"
            return

        if yuv1.file_size % (block_size ** 2) != 0 or yuv2.file_size % (block_size ** 2) != 0:
            print "data size: frame1 {0},  frame2 {1} are not divisible by blocksize {2} ".format(yuv1.file_size,
                                                                                                  yuv2.file_size,
                                                                                                  block_size)
            return

        # create the output mse array
        if yuv1.bytes_per_sample == 1:
            mse_array = array.array('B')
        elif yuv1.bytes_per_sample == 2:
            mse_array = array.array('H')

        # only take the y channel
        mse_array = yuv1.YUV_data[:yuv1.luma_size]
        print "the original mse_array size is {0}".format(len(mse_array))
        # left to right and up to bottom, generate a y' with mse
        w = yuv1.YUV_width
        h = yuv1.YUV_height
        w_blocks = w / block_size
        h_blocks = h / block_size

        for w_j in xrange(0, w_blocks):
            for h_i in xrange(0, h_blocks):
                block1 = []
                block2 = []
                for k in range(0, block_size):
                    startaddr = (h_i * block_size + k) * w + w_j * block_size
                    block1.extend(yuv1.YUV_data[startaddr: (startaddr + block_size)])
                    block2.extend(yuv2.YUV_data[startaddr: (startaddr + block_size)])
                ##################
                # mse analysis
                ##################
                mse = self.MSE(block1, block2)

                # assign the mse to the array
                for h_k in range(0, block_size):
                    startaddr = (h_i * block_size + h_k) * w + w_j * block_size
                    for block_k in range(0, block_size):
                        mse_array[startaddr + block_k] = int(mse*5)

        # create a mse array and return it
        print "the size of mse_array is {0}".format(len(mse_array))
        print "#### end of frame_diff_by_block"
        return mse_array


def main(argv):
    # print "start test"
    # test_block_mse()
    # return

    parser = argparse.ArgumentParser(description="""
    visual quality metrix tool for YUV files.
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
    # optional args
    parser.add_argument(
        '-input_folder',
        action='store',
        dest='input_folder',
        default='',
        help=' all the files with .yuv extension in the given directory  will be added for comparison. TBD......'
    )

    parser.add_argument(
        '-yuvfiles',
        action='store',
        dest='yuv_files',
        default=[],
        nargs='+',
        help=' all the YUV files for comparison.'
    )

    parser.add_argument(
        '-multireport',
        action='store_true',
        dest='multireport',
        default='False',
        help='Each YUV file has its own CSV report. TBD......'
    )

    parser.add_argument(
        '-mse_blocksize',
        action='store',
        dest='mse_blocksize',
        type=int,
        default=0,
        help='''
        When this option is specified  a new yuv file will be generated. It''s y
        channel will be the mse value for each block.'''
    )
    # parse the arguments for the current command line.
    opt = parser.parse_args()

    # validate the input
    if opt.input_folder == '' and len(opt.yuv_files) == 0:
        print "empty input. Please at least specify either -input_folder or -yuvfiles."
        return 1
    elif os.path.exists(opt.input_folder):
        # find all the yuv files under -input_folder and append it to the opt.yuv_files
        print "adding yuv files under {0} to the list.".format(opt.input_folder)
        opt.yuv_files += glob.glob('{0}/*.yuv'.format(opt.input_folder))

    else:
        print "input_folder is invalid. Only YUV files listed will be computed."

    # remove duplications
    yuvfile_set = set()
    for item in opt.yuv_files:
        # if item in yuvfile_set: continue
        yuvfile_set.add(os.path.normpath(item))

    # Setup YUV reader

    YUVs = set()
    for yuv_file in yuvfile_set:
        YUVs.add(YUVFile(yuv_file, opt.yuv_format, opt.width, opt.height))

    if opt.mse_blocksize == 0:

        # create header of csv
        # Print csv_header
        # run PSNR on the YUV_reader,

        csvdata = []
        csv_header = ['#']
        for yuv in YUVs:

            csv_header.append(yuv.YUV_file)
            # add CSV row
            csvrow = [yuv.YUV_file]
            for second_yuv in YUVs:
                if yuv is second_yuv:
                    psnr_res = 100
                else:
                    VQM_func = VQM()
                    print "{0} vs {1} \n".format(yuv.YUV_file, second_yuv.YUV_file)
                    mse_res, psnr_res_all, psnr_res = VQM_func.frame_evaluation(yuv.YUV_data, second_yuv.YUV_data)
                    print "PSNR = {0}".format(psnr_res)
                csvrow.append(psnr_res)

            print csvrow
            csvdata.append(csvrow)

        with open('report.csv', 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(csv_header)
            for row in csvdata:
                csvwriter.writerow(row)
    else:
        # do the mse_block evaluation
        VQM_func = VQM()

        # James: should not use list since it will involove a sorting
        # list(YUVs) will put the last added it as the first

        YUVs_list = list(YUVs)
        if YUVs_list[0].bytes_per_sample == 1:
            mse_y_array = array.array('B')
        elif YUVs_list[0].bytes_per_sample == 2:
            mse_y_array = array.array('H')

        # mse for unpatched frames
        print "firt yuv is {0} second is {1}".format(YUVs_list[0].YUV_file, YUVs_list[1].YUV_file)
        mse_y_array = VQM_func.frame_diff_by_block(YUVs_list[0].YUV_data, YUVs_list[1].YUV_data, opt.mse_blocksize)

        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(YUVs_list[0].YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of extended mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))
            with open("mse.yuv", 'wb') as mse_file:
                # mse_y_array.tofile(mse_file)
                mse_file.write(mse_y_array)


if __name__ == '__main__':
    main(sys.argv)

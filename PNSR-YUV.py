import array
import sys
import os
import glob
import math
import re
import argparse
import copy
import csv
import logging


__author__ = 'James F'
__version__ = '0.0.1'

# create logger with 'spam_application'
logger = logging.getLogger('Patch_YUV')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
os.remove('debug.log')
fh = logging.FileHandler('debug.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# yuv format
# forurcc, sample format, channel order, Interlace/Progressive, packed/plannar


class YUV_File:
    """This is a YUV parser to read YUV in different formats.
        for each YUV file give the start position of Y U V
        Current YUV formats are defined in http://www.fourcc.org/yuv.php.
        yv12: 420 planar, Y, V, U
        iyuv, i420: 420 planar, Y, U, V
        nv12: 420 semi-planar, Y, UVUV
        nv21: 420 semi-planar, Y, VUVU
        yuv2, yuyv: 422 packed, YUYV
        uyvy: 422 packed, UYVY
        yuyv: 422 packed, YUYV
    """

    # fourcc format dict,
    # fourcc name,     planar/packed,y,uvx,uvy, interlace/progressive
    #  bytes_per_sample is determined by the file size and format

    yuv_format_dict = {'i420': ['planar', 4, 2, 2, 'progressive'],
                       'yuv422p10le': ['planar', 4, 2, 4, 'progressive'],
                       'iyuv': ['planar', 4, 2, 2, 'progressive'],
                       'yv12': ['planar', 4, 2, 2, 'progressive'],
                       'ayuv': ['planar', 4, 4, 4, 'progressive'],
                       'yuva444': ['planar', 4, 4, 4, 'progressive'],
                       'yuv422': ['planar', 4, 2, 4, 'progressive']
                       }

    def __init__(self, yuv_file, yuv_format, yuv_width, yuv_height):
        # supported_formats = ('yv12', 'iyuv', 'i420', 'nv12', 'nv21')
        self.supported_formats = {'yv12', 'iyuv', 'i420', 'ayuv', 'yuva444', 'yuv422', 'yuv422p10le'}

        # the YUV file
        self.YUV_file = yuv_file
        # YUV format
        self.YUV_format = yuv_format
        self.YUV_width = yuv_width
        self.YUV_height = yuv_height
        self.bytes_per_sample = 1

        self.begin = 0
        print yuv_file
        file_size = os.path.getsize(yuv_file)
        self.end = file_size

        # verify the format is supported
        # bits is 8,10,12,14,16, They all use 2 bytes
        if self.YUV_format.lower() not in self.supported_formats:
            print "{0} is not supported. Only following types {1} are supported.".format(yuv_format,
                                                                                         list(self.supported_formats))
            sys.exit(1)

        coffin = 1.0 + self.yuv_format_dict[yuv_format.lower()][2] * self.yuv_format_dict[yuv_format.lower()][
            3] * 2.0 / (self.yuv_format_dict[yuv_format.lower()][1] ** 2)
        #        if yuv_width * yuv_height * 3 / 2 <= file_size:
        # print "y {0} , u {1} , v {2}coffin is {3}, file_size {4} \n".format(self.yuv_format_dict[yuv_format.lower()][1], self.yuv_format_dict[yuv_format.lower()][2], self.yuv_format_dict[yuv_format.lower()][3], coffin, file_size)

        if yuv_width * yuv_height * coffin == file_size:
            self.bytes_per_sample = 1
        elif yuv_width * yuv_height * coffin * 2 == file_size:
            self.bytes_per_sample = 2
        else:
            print "wrong width {0} and height {1} setting or wrong file size {2}.".format(yuv_width, yuv_height,
                                                                                          file_size)
            sys.exit(1)

        # YUV_data
        # if its 8 bit
        if self.bytes_per_sample == 1:
            self.YUV_data = array.array('B')
            # if its 10-16 bit
        elif self.bytes_per_sample == 2:
            self.YUV_data = array.array('H')

            # read entire file
        with open(yuv_file, 'rb') as content_file:
            self.YUV_data.fromfile(content_file, file_size / self.bytes_per_sample)
        print "the yuv_data size is {0}".format(len(self.YUV_data))
        # print "printting YUVDATA \n"
        # print self.YUV_data
        # if file1_size != file2_size:
        #        print "warning, file sizes do not match! comparing min size %d bytes" % minsize

        #    if len(argv) >= 5:
        #        w = int(argv[3])
        #        h = int(argv[4])
        #    else:
        #        try:
        #            w, h = width_height_from_str(filename1)
        #        except RuntimeError:
        #            try:
        #                w, h = width_height_from_str(filename2)
        #            except RuntimeError:
        #                print "failed to parse width,height from filename"
        #                usage(argv[0])
        #                return

        #    assert w * h * 3 / 2 <= minsize
        #    data_end = w * h * 3 / 2

        #    data1.fromfile(open(filename1, "rb"), minsize)
        #    data2.fromfile(open(filename2, "rb"), minsize)

        luma_size = yuv_width * yuv_height
        # for YUV420
        if self.YUV_format == 'i420':
            chroma_size = yuv_width * yuv_height / 4
            frame_size = yuv_width * yuv_height * 3 / 2

        elif self.YUV_format == 'yuv422p10le':
            chroma_size = yuv_width * yuv_height / 2
            frame_size = yuv_width * yuv_height * 2

        print "data size: %d, luma=%d, chroma=%d, frame=%d" % (file_size,
                                                               luma_size, chroma_size, frame_size)
        # start of Y U V
        y = 0
        u = y + luma_size
        v = u + chroma_size

        # self.begins = [y, u, v, y]
        # self.ends = [u, v, end_pos, end_pos]
        # self.sizes = [luma_size, chroma_size, chroma_size, frame_size]

        self.yuv = {'y': {y, u, luma_size}, 'u': {u, v, chroma_size}, 'v': {v, file_size, chroma_size},
                    'frame': {y, file_size, frame_size}}
        # return data, begin, end, size


class PNSR:
    """compute PNSR for frames.
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

    def sum_square_err(self, data1, data2, beg, end):
        return sum((a - b) * (a - b) for a, b in zip(data1[beg:end], data2[beg:end]))

    def psnr(self, mse):
        log10 = math.log10
        if mse == 0:
            mse = sys.float_info.min
        return 10.0 * log10(float(256 * 256) / float(mse))

    def frame_diff(self, frame1_data, frame2_data, begin_pos, end_pos, w, h):
        luma_size = w * h
        chroma_size = w * h / 4
        frame_size = w * h * 3 / 2

        print "data size: %d, w*h=%d" % (end_pos - begin_pos,
                                         luma_size)
        print "evaluating mse.."

        # start of Y U V
        y = begin_pos
        u = y + luma_size
        v = u + chroma_size

        begin = [y, u, v, y]
        end = [u, v, end_pos, end_pos]
        size = [luma_size, chroma_size, chroma_size, frame_size]

        colorspace_mse = [self.sum_square_err(frame1_data, frame2_data, begin[i], end[i]) / float(size[i]) for i in
                          range(4)]

        colorspace_psnr = [self.psnr(m) for m in colorspace_mse]
        return colorspace_mse, colorspace_psnr, colorspace_psnr[-1]

    def frames_diff(self, frames):
        for frame_i in frames:
            for frame_j in frames:
                print frame_i.YUV_data
                colorplane_mse, colorplane_psnr, frame_psnr = frame_diff(frame_i, frame_j, begin_pos, end_pos, w, h)

    # output to different files for each metrix, mse, psnr, psnr_y,psnr_u, psnr_v
    def output_csv(self, pnsr_list):
        return

    def width_height_from_str(self, s):
        m = re.search(".*[_-](\d+)x(\d+).*", s)
        if not m:
            raise RuntimeError()

        w = int(m.group(1))
        h = int(m.group(2))
        return w, h

    # compute the mse for two same size data
    def frame_diff_by_block(self, frame1_data, frame2_data, w, h, bytes_per_sample, blocksize):

        # validation, the size of frame_data should dividable by blocksize**2
        if (len(frame1_data) != len(frame2_data)):
            print "the data size is not the same"
            return

        if (len(frame1_data) % (blocksize ** 2) != 0 or len(frame2_data) % (blocksize ** 2) != 0):
            print "data size: frame1 {0},  frame2 {1} are not divisible by blocksize {2} ".format(len(frame1_data),
                                                                                                  len(frame2_data),
                                                                                                  blocksize)
            return

        # create the output mse array
        if bytes_per_sample == 1:
            mse_array = array.array('B')
        elif bytes_per_sample == 2:
            mse_array = array.array('H')

        # only take the y channel
        mse_array = frame1_data[:w * h]
        print "the original mse_array size is {0}".format(len(mse_array))
        # left to right and up to bottom, generate a y' with mse
        w_blocks = w / blocksize
        h_blocks = h / blocksize


        for w_j in xrange(0, w_blocks):
            for h_i in xrange(0, h_blocks):
                block1 = []
                block2 = []
                for k in range(0, blocksize):
                    startaddr = (h_i * blocksize + k) * w + w_j * blocksize
                    block1.extend(frame1_data[startaddr: (startaddr + blocksize)])
                    block2.extend(frame2_data[startaddr: (startaddr + blocksize)])
                ##################
                # mse analysis
                ##################
                mse = self.sum_square_err(block1, block2, 0, blocksize**2) / float(blocksize**2)

                # assign the mse to the array
                for h_k in range(0, blocksize):
                    startaddr = (h_i * blocksize + h_k) * w + w_j * blocksize
                    for block_k in range(0, blocksize):
                        mse_array[startaddr + block_k] = int(mse*5)


        # create a mse array and return it
        print "the size of mse_array is {0}".format(len(mse_array))
        print "#### end of frame_diff_by_block"
        return mse_array


    # patching the watermark
    def patching_frame(self, frame_data_wt, frame_data_org, w, h, bytes_per_sample, blocksize):
        log_block_x = 64
        log_block_y = 55
        # validation, the size of frame_data should dividable by blocksize**2
        if (len(frame_data_wt) != len(frame_data_org)):
            print "the data size is not the same"
            return

        if (len(frame_data_wt) % (blocksize ** 2) != 0 or len(frame_data_org) % (blocksize ** 2) != 0):
            print "data size: frame1 {0},  frame2 {1} are not divisible by blocksize {2} ".format(len(frame_data_wt),
                                                                                                  len(frame_data_org),
                                                                                                  blocksize)
            return

        # create the output mse array
        if bytes_per_sample == 1:
            mse_array = array.array('B')
        elif bytes_per_sample == 2:
            mse_array = array.array('H')

        # only take the y channel
        mse_array = frame_data_org[:w * h]
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
                    block_wt.extend(frame_data_wt[startaddr: (startaddr + blocksize)])
                    block_org.extend(frame_data_org[startaddr: (startaddr + blocksize)])

                #if w_j == log_block_x and h_i == log_block_y:
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

                AD = H_Diff - L_Diff
                # max AD , 10, AD/max_AD to scale down the changes
                # show we only scale down the change on a pixel or entire block?
                # compute the avg of block. if it is quite even, we should use a lower MAX_AD
                # Otherwise we should use a higher MAX_AD
                # MAX_L_H_AD = 8.0
                MAX_L_H_AD = average_diversity
                # should be adjusted by average diversity
                logger.debug(" block {0} {1}".format(w_j, h_i))
                logger.debug("[AD, MAX_L_H_AD] is [{0},{1}]".format(AD, MAX_L_H_AD))

                if AD > MAX_L_H_AD:
                    # print "original block : \n"
                    # print block1

                    # print "patched block : \n"

                    scaledown_blocks += 1
                    # we need scale down the watermark
                    # assign the mse to the array
                    for h_k in range(0, blocksize):
                        startaddr = (h_i * blocksize + h_k) * w + w_j * blocksize
                        for block_k in range(0, blocksize):
                            pixel = block_org[h_k * blocksize + block_k]
                            mse_array[startaddr + block_k] = int(
                                pixel + (block_wt[h_k * blocksize + block_k] - pixel) * (MAX_L_H_AD * 1.0 / AD))
                            #if w_j == 26 and h_i == 61:
                             #   print " pixel {0}, change {1}, result {2}".format(pixel, block2[h_k * blocksize + block_k] - pixel, mse_array[startaddr + block_k])
                        #if w_j == log_block_x and h_i == log_block_y:
                        logger.debug(mse_array[startaddr: startaddr + blocksize])

# mse_array[startaddr + block_k] = block2[h_k*blocksize+block_k]
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


def test_block_mse():
    frame1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    frame2 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, ]
    PNSR_func = PNSR()
    print PNSR_func.frame_diff_by_block(frame1, frame2, 16, 16, 1, 8)


def main(argv):
    # print "start test"
    # test_block_mse()
    # return


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
    # optional args
    parser.add_argument(
        '-input_folder',
        action='store',
        dest='input_folder',
        default='',
        help=' all the files with .yuv extension in the given directory  will be added for comparison. No implemented.'
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
        help='Each YUV file has its own CSV report.'
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
        YUVs.add(YUV_File(yuv_file, opt.yuv_format, opt.width, opt.height))

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
                    PNSR_func = PNSR()
                    print "{0} vs {1} \n".format(yuv.YUV_file, second_yuv.YUV_file)
                    mse_res, psnr_res_all, psnr_res = PNSR_func.frame_diff(yuv.YUV_data, second_yuv.YUV_data, yuv.begin,
                                                                           yuv.end, opt.width, opt.height)
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
        PNSR_func = PNSR()

        # James: should not use list since it will involove a sorting
        # list(YUVs) will put the last added it as the first

        YUVs_list = list(YUVs)
        if YUVs_list[0].bytes_per_sample == 1:
            mse_y_array = array.array('B')
        elif YUVs_list[0].bytes_per_sample == 2:
            mse_y_array = array.array('H')

        # mse for unpatched frames
        print "firt yuv is {0} second is {1}".format(YUVs_list[0].YUV_file, YUVs_list[1].YUV_file,)
        mse_y_array = PNSR_func.frame_diff_by_block(YUVs_list[0].YUV_data, YUVs_list[1].YUV_data, opt.width, opt.height,
                                               YUVs_list[0].bytes_per_sample, opt.mse_blocksize)


        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(YUVs_list[0].YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))
            with open("mse.yuv", 'wb') as mse_file:
                # mse_y_array.tofile(mse_file)
                mse_file.write(mse_y_array)


        #start patching

        print "the watermarked file (YUVs_list[0].YUV_data:{0}".format(YUVs_list[0].YUV_file)
        print "the orginal file YUVs_list[1].YUV_data: {0}".format(YUVs_list[1].YUV_file)
        mse_y_array = PNSR_func.patching_frame(YUVs_list[0].YUV_data, YUVs_list[1].YUV_data, opt.width, opt.height,
                                                    YUVs_list[1].bytes_per_sample, opt.mse_blocksize)
        if opt.yuv_format == 'yuv422p10le':
            mse_y_array.extend(YUVs_list[1].YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))
            with open("patched_frame.yuv", 'wb') as mse_file:
                # mse_y_array.tofile(mse_file)
                mse_file.write(mse_y_array)

            # we generate a mse_pathed.yuv for testing purpose
            patched_yuv = YUV_File("patched_frame.yuv", opt.yuv_format, opt.width, opt.height)
            patched_mse_y_array = PNSR_func.frame_diff_by_block(YUVs_list[0].YUV_data, patched_yuv.YUV_data, opt.width,
                                                                opt.height, YUVs_list[1].bytes_per_sample, opt.mse_blocksize)


            patched_mse_y_array.extend(YUVs_list[1].YUV_data[opt.width * opt.height:])
            print "the array type of mse_y_array: {0} and the size of mse_y_array is {1}".format(mse_y_array.typecode,
                                                                                                 len(mse_y_array))
            with open("patched_mse.yuv", 'wb') as mse_file:
                # mse_y_array.tofile(mse_file)
                mse_file.write(patched_mse_y_array)

if __name__ == '__main__':
    main(sys.argv)

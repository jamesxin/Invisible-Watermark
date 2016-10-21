import array
import sys
import os
import glob
import re
import argparse


__author__ = 'James F'
__version__ = '0.0.1'


class YUVFile:
    """This is a YUV parser to read YUV in different formats.
        for each YUV file give the start position of Y U V
        Current YUV formats are defined in http://www.fourcc.org/YUV.php.
        yv12: 420 planar, Y, V, U
        iYUV, i420: 420 planar, Y, U, V
        nv12: 420 semi-planar, Y, UVUV
        nv21: 420 semi-planar, Y, VUVU
        YUV2, yuyv: 422 packed, YUYV
        uyvy: 422 packed, UYVY
        yuyv: 422 packed, YUYV
    """

    # YUV format
    # forurcc, sample format, channel order, Interlace/Progressive, packed/plannar
    # fourcc format dict,
    # fourcc name,     planar/packed,y,u_x,u_y, interlace/progressive
    #  bytes_per_sample is determined by the file size and format

    YUV_format_dict = {'i420':          ['planar', 4, 2, 2, 'progressive'],
                       'iyuv':          ['planar', 4, 2, 2, 'progressive'],
                       'yv12':          ['planar', 4, 2, 2, 'progressive'],
                       'ayuv':          ['planar', 4, 4, 4, 'progressive'],
                       'yuv444':        ['planar', 4, 4, 4, 'progressive'],
                       'yuv422p10le':   ['planar', 4, 2, 4, 'progressive'],
                       'yuv422':        ['planar', 4, 2, 4, 'progressive']
                       }

    def __init__(self, YUV_file, YUV_format, YUV_width, YUV_height):
        self.supported_formats = {'yv12', 'iyuv', 'i420', 'ayuv', 'yuv444', 'yuv422', 'yuv422p10le'}

        # the YUV file
        self.YUV_file = YUV_file

        # YUV format
        self.YUV_format = YUV_format
        self.YUV_width = YUV_width
        self.YUV_height = YUV_height
        self.bytes_per_sample = 1

        self.begin = 0
        self.file_size = os.path.getsize(YUV_file)
        self.end = self.file_size

        # verify the format is supported
        # bits is 8,10,12,14,16, They all use 2 bytes

        if self.YUV_format.lower() not in self.supported_formats:
            print "{0} is not supported. Only following types {1} are supported.".format(YUV_format,
                                                                                         list(self.supported_formats))
            sys.exit(1)

        factor = 1.0 + self.YUV_format_dict[YUV_format.lower()][2] * self.YUV_format_dict[YUV_format.lower()][
            3] * 2.0 / (self.YUV_format_dict[YUV_format.lower()][1] ** 2)

        if YUV_width * YUV_height * factor == self.file_size:
            self.bytes_per_sample = 1
        elif YUV_width * YUV_height * factor * 2 == self.file_size:
            self.bytes_per_sample = 2
        else:
            print "wrong width {0} and height {1} setting or wrong file size {2}.".format(YUV_width, YUV_height,
                                                                                          self.file_size)
            sys.exit(1)

        # YUV_data
        # 8 bit
        if self.bytes_per_sample == 1:
            self.YUV_data = array.array('B')

        # 10-16 bit
        elif self.bytes_per_sample == 2:
            self.YUV_data = array.array('H')

        # read entire file
        with open(YUV_file, 'rb') as content_file:
            self.YUV_data.fromfile(content_file, self.file_size / self.bytes_per_sample)
        print "the YUV_data size is {0}".format(len(self.YUV_data))

        self.luma_size = YUV_width * YUV_height
        chroma_ratio = 1.0 * self.YUV_format_dict[YUV_format.lower()][2] / self.YUV_format_dict[YUV_format.lower()][1] * \
                       1.0 * self.YUV_format_dict[YUV_format.lower()][3] / self.YUV_format_dict[YUV_format.lower()][1]
        self.chroma_size = self.luma_size * chroma_ratio
        self.frame_size = self.luma_size + self.chroma_size*2

        print "data size: %d, luma=%d, chroma=%d, frame=%d" % (self.file_size,
                                                               self.luma_size, self.chroma_size, self.frame_size)
        # start of Y U V
        y = 0
        u = y + self.luma_size
        v = u + self.chroma_size

        # self.begins = [y, u, v, y]
        # self.ends = [u, v, end_pos, end_pos]
        # self.sizes = [luma_size, chroma_size, chroma_size, frame_size]

        self.YUV = {'y': {y, u, self.luma_size}, 'u': {u, v, self.chroma_size}, 'v': {v, self.file_size, self.chroma_size},
                    'frame': {y, self.file_size, self.frame_size}}
        # return data, begin, end, size

    # parsing w,h from the filename
    def width_height_from_str(self, s):
        m = re.search(".*[_-](\d+)x(\d+).*", s)
        if not m:
            raise RuntimeError()
    
        w = int(m.group(1))
        h = int(m.group(2))
        return w, h


def main(argv):
    parser = argparse.ArgumentParser(description="""
    YUV lib, the byte_per_sample will be autometically determined by file size, YUV format, width and height.
    """)

    # set the program version
    parser.add_argument('-version', action='version', version='%(prog)s {0}'.format(__version__))

    # specify the input parameters

    # required args
    parser.add_argument(
        '-format',
        action='store',
        dest='YUV_format',
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
        help=' all the files with .YUV extension in the given directory  will be added for comparison.'
    )

    parser.add_argument(
        '-YUV_files',
        action='store',
        dest='YUV_files',
        default=[],
        nargs='+',
        help=' all the YUV files for comparison.'
    )
    # parse the arguments for the current command line.
    opt = parser.parse_args()

    # validate the input
    if opt.input_folder == '' and len(opt.YUV_files) == 0:
        #print "empty input. Please at least specify either -input_folder or -YUVfiles."
        #return 1
        print "No input_folder and yuv_files have been entered. Searching current directory for .yuv files."
        opt.input_folder = '.'

    if os.path.exists(opt.input_folder):
        # find all the YUV files under -input_folder and append it to the opt.YUV_files
        print "adding YUV files under {0} to the list.".format(opt.input_folder)
        opt.YUV_files += glob.glob('{0}/*.yuv'.format(opt.input_folder))

    else:
        print "input_folder is invalid. Only YUV files listed will be computed."
    
    # remove duplications
    YUV_file_set = set()
    for item in opt.YUV_files:
        if os.path.exists(item):
            YUV_file_set.add(os.path.normpath(item))
        else:
            print "Ignore {0} since it does't exist.".format(item)
    
    YUVs = set()
    for YUV_file in YUV_file_set:
        YUVs.add(YUVFile(YUV_file, opt.YUV_format, opt.width, opt.height))

if __name__ == '__main__':
    main(sys.argv)

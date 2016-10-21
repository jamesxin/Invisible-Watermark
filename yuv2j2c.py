__author__ = 'jfei'
__version__ = '0.1'

import sys
import os
import glob
import argparse
import subprocess


def main(argv):
    parser = argparse.ArgumentParser(description="""
    convert wm1 files to j2c files.
    """)

    # set the program version
    parser.add_argument('-version', action='version', version='{0}s {1}'.format(__file__,__version__))

    # specify the input parameters

    # required args
    parser.add_argument(
        '-input_dir',
        action='store',
        dest='input_dir',
        default='.',
        help=' The input directory.'
    )

    opt = parser.parse_args()

    files = glob.glob('{0}/*.j2k.wm1'.format(opt.input_dir))
    for yuvfile in files:
            target_filename = os.path.splitext(os.path.splitext(yuvfile)[0])[0]+'.j2c'
            command_line = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "yuv422p10le",
                "-s", "3840x2160",
                "-i", yuvfile,
                "-vcodec", "libopenjpeg",
                "-q", "0",
                "-compression_level", "5",
                "-format", "j2k",
                "-profile", "cinema4k",
                target_filename
            ]
            print "Command: {0}".format(command_line)
            process = subprocess.Popen(command_line)
            #process.communicate()
            ret = process.wait()
            print "Command return: {0}".format(ret)


if __name__ == '__main__':
    main(sys.argv)

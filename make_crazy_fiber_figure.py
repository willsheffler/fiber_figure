import pymol
from pymol import cmd
import sys


def main():
    pymol.finish_launching()

    cmd.load(sys.argv[1], 'orig')

if __name__ == '__main__':
    main()

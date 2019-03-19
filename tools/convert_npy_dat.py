#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import struct

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input", action='store', help="Input file.", type=str)

    return parser.parse_args()


def convert(args):
    arr = np.load(args.input)
    if arr.ndim != 2:
        raise ValueError("Input should be bidimensional.")
    
    out_path = os.path.splitext(args.input)[0] + ".dat"
    
    rows, cols = arr.shape
    print(rows, cols)
    rav = arr.ravel()

    header = struct.pack('ii', int(rows), int(cols))
    data = struct.pack('f'*rav.shape[0], *rav)

    with open(out_path, "wb") as f:
        f.write(header+data)


def main():
    args = parse_args()
    convert(args)



if __name__ == "__main__":
    main()



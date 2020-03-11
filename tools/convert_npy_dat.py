#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ******************************************************
# * This script converts .npy files to a .dat readable *
# * by read_mat and read_mat_labels in util.hpp.       *
# *                                                    *
# * Author: Thierry Moreira                            *
# *                                                    *
# ******************************************************
#
# Copyright 2019 Thierry Moreira
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import struct

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input", action='store', help="Input file.", type=str)
    parser.add_argument("-l", "--labels", action='store_true', help="Input file.")

    return parser.parse_args()


def npy_dat(args):
    arr = np.load(args.input)
    if arr.ndim != 2:
        raise ValueError("Input should be bidimensional.")
    
    out_path = os.path.splitext(args.input)[0] + ".dat"
    
    rows, cols = arr.shape
    if args.labels:
        cols -= 1
    print(rows, cols)

    header = struct.pack('ii', int(rows), int(cols))
    data = []
    if args.labels:
        for row in arr:
            data.append(struct.pack('i', int(row[0])))
            data.append(struct.pack('f'*(row.shape[0]-1), *row[1:]))
    else:
        rav = arr.ravel()
        data = data.append(struct.pack('f'*rav.shape[0], *rav))

    with open(out_path, "wb") as f:
        f.write(header)
        for d in data:
            f.write(d)

def dat_npy(args):
    arr = np.load(args.input)
    if arr.ndim != 2:
        raise ValueError("Input should be bidimensional.")
    
    out_path = os.path.splitext(args.input)[0] + ".dat"
    
    rows, cols = arr.shape
    if args.labels:
        cols -= 1
    print(rows, cols)

    header = struct.pack('ii', int(rows), int(cols))
    data = []
    if args.labels:
        for row in arr:
            data.append(struct.pack('i', int(row[0])))
            data.append(struct.pack('f'*(row.shape[0]-1), *row[1:]))
    else:
        rav = arr.ravel()
        data = data.append(struct.pack('f'*rav.shape[0], *rav))

    with open(out_path, "wb") as f:
        f.write(header)
        for d in data:
            f.write(d)


def main():
    args = parse_args()
    npy_dat(args)



if __name__ == "__main__":
    main()



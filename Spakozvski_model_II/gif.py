#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:35:54 2023

@author: fneri
routine to make a gif
"""

from PIL import Image
import os

# set the path for the images
path = "pics/pics_gif/"

# get all image filenames in the directory
files = os.listdir(path)

# sort the files to ensure proper ordering
files.sort()

# create a list of image objects from the filenames
images = [Image.open(os.path.join(path, f)) for f in files]

# create a GIF file from the image objects
images[0].save('pics/poles_evolution.gif', save_all=True, append_images=images[1:], duration=700, loop=0)


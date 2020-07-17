import numpy as np
from PIL import Image
import os
import sys
import argparse

PATH = '..'
GAP = 10
SIZE = 512

old = "pre-RW"
namesN = ["img", "FT", "LWF", "ILT", "IC", "SYM", "tar"]
names0 = ["img", old, old, old, old, old, "tar"]


def main(opts):
    step = opts.step
    path = PATH
    dest = PATH + "/cat"
    names = names0 if step == 0 else namesN

    os.makedirs(dest, exist_ok=True)

    for i in range(600):
        new_im = Image.new('RGB', (SIZE * len(names) + GAP * (len(names)-1),
                                   GAP * (step - 1) + step * SIZE), (255, 255, 255, 255))
        for s in range(step):
            if s == 0:
                names = names0
            else:
                names = namesN
            fol_path = path + "/" + str(s) + "/"
            for j, n in enumerate(names):
                im = Image.open(fol_path+f"0-{i}/"+n+f".png")
                im.thumbnail((SIZE, SIZE))
                new_im.paste(im, (j*(GAP+SIZE), s*(SIZE+GAP)))

            new_im.save(dest+f"/{i}.png")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default=1, type=int)

    opts = parser.parse_args()
    main(opts)

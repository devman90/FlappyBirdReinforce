import os
import subprocess as sp

import cv2
import torch
import numpy as np


def init_weights(m):
    if type(m) == torch.nn.Conv2d  or type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.)
        # torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


class Recorder:
    def __init__(self, video_filename, fps=30, width=288, height=512):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.filename = video_filename
        self.fps = fps
        self.width = width
        self.height = height

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(
            self.filename,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

    def write(self, image):
        image = np.transpose(image, (1,0,2))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.normalize(image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.out.write(image)

    def write_score(self, score, runtime):
        image = np.zeros((self.height, self.width, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 200)
        fontScale = 0.6
        color = (240, 125, 0)
        thickness = 2
        im_with_text = cv2.putText(image,
                                   "Final Score: {}".format(score),
                                   org,
                                   font,
                                   fontScale,
                                   color,
                                   thickness,
                                   cv2.LINE_AA
                                   )
        im_with_text = cv2.putText(image,
                                   "Total Run Time: {:.3f}".format(runtime),
                                   (20,250),
                                   font,
                                   fontScale,
                                   color,
                                   thickness,
                                   cv2.LINE_AA
                                   )
        for _ in range(20):
            self.out.write(im_with_text)

    def save(self):
        self.out.release()
        # Convert from AVI to MP4
        sp.run(['ffmpeg', '-y',
                '-loglevel', '16',
                '-i', '{}'.format(self.filename),
                '{}'.format(self.filename[:-4] + '.mp4')
                ])
        os.remove(self.filename)

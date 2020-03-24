import numpy as np
from cv2 import line, resize, INTER_CUBIC
import json
import cairocffi as cairo
from flags import FLAGS


class Sketch(object):

    def __init__(self,  canvas_shape=(256, 256),
                 dtype=np.uint8,
                 thickness=6,
                 image_shape=FLAGS.image_shape):

        super(Sketch, self).__init__()

        self.canvas_shape = canvas_shape
        self.dtype = dtype
        self.thickness = thickness
        self.image_shape = image_shape

    def __call__(self, sample):

        drawing = sample
        strokes = json.loads(drawing)
        image = np.zeros(self.canvas_shape, self.dtype)

        for t, stroke in enumerate(strokes):
            color = 255 - min(t, 10) * 10
            for i in range(len(stroke[0]) - 1):
                _ = line(image,
                         (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]),
                         color,
                         self.thickness)
        sample = resize(image, self.image_shape, interpolation=INTER_CUBIC)
        sample = sample.reshape(self.image_shape + (1,))
        return sample


class Draw(object):

    def __init__(self, side=FLAGS.image_shape[0],
                 thickness=6,
                 padding=6,
                 bg_color=(0, 0, 0),
                 fg_color=(1, 1, 1)):

        super(Draw, self).__init__()
        self.side = side
        self.thickness = thickness
        self.padding = padding
        self.bg_color = bg_color
        self.fg_color = fg_color

    def __call__(self, sample):

        original_side = 256.
        drawing = sample
        strokes = json.loads(drawing)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.side, self.side)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(self.thickness)

        # scale to match the new size
        # add padding at the edges for the thickness
        # and add additional padding to account for anti aliasing
        total_padding = self.padding * 2. + self.thickness
        new_scale = float(self.side) / float(original_side + total_padding)
        ctx.scale(new_scale, new_scale)
        ctx.translate(total_padding / 2., total_padding / 2.)

        # clear background
        ctx.set_source_rgb(*self.bg_color)
        ctx.paint()

        bbox = np.hstack(strokes).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in strokes]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*self.fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        sample = np.array(data)[::4].reshape((self.side, self.side) + (1,))
        return sample

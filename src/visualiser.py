# File adapted from https://github.com/gml16/rl-medical/blob/master/src/viewer.py

import math

import numpy as np
import pyglet
from pyglet.gl import (
    gl,
    glTexParameteri,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_TEXTURE_MIN_FILTER,
    glScalef,
    glEnable,
    GL_BLEND,
    glBlendFunc,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GLubyte,
    glBegin,
    GL_POINTS,
    glVertex3f,
    glEnd,
    GL_QUADS,
    GL_POLYGON,
    GL_TRIANGLES,
    glColor4f,
)


class SimpleImageViewer:
    """Simple image viewer class for rendering images using pyglet"""

    def __init__(self, height, width, scale_x=1, scale_y=1, caption=None):
        self.window = pyglet.window.Window(
            width=scale_x * width,
            height=scale_y * height,
            caption=caption,
            resizable=True,
        )
        location_x = 100
        location_y = 100
        self.window.set_location(location_x, location_y)
        self.window.flip()

        # scale window size
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glScalef(scale_x, scale_y, 1.0)

        self.img_height = height

        # turn on transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_image(self, img: np.ndarray) -> None:
        # convert data type to GLubyte
        raw_data = (GLubyte * img.size)(*list(img.ravel().astype("int")))
        image = pyglet.image.ImageData(img.shape[0], img.shape[1], "RGB", raw_data, pitch=img.shape[1] * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)

    def draw_point(self, x=0.0, y=0.0, z=0.0):
        x = self.img_height - x
        glBegin(GL_POINTS)  # draw point
        glVertex3f(x, y, z)
        glEnd()

    def draw_circle(self, radius=10, res=30, pos_x=0, pos_y=0, color=(1.0, 1.0, 1.0, 1.0), **attrs):
        points = []
        # window start indexing from bottom left
        x = self.img_height - pos_x
        y = pos_y

        for i in range(res):
            ang = 2 * math.pi * i / res
            points.append((math.cos(ang) * radius + y, math.sin(ang) * radius + x))

        # draw filled polygon
        if len(points) == 4:
            glBegin(GL_QUADS)
        elif len(points) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in points:
            # choose color
            glColor4f(color[0], color[1], color[2], color[3])
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()
        # reset color
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def draw_rect(self, x_min_init, y_min, x_max_init, y_max):
        main_batch = pyglet.graphics.Batch()
        # fix location
        x_max = self.img_height - x_max_init
        x_min = self.img_height - x_min_init
        # draw lines
        glColor4f(0.8, 0.8, 0.0, 1.0)
        main_batch.add(2, gl.GL_LINES, None, ("v2f", (y_min, x_min, y_max, x_min)))
        main_batch.add(2, gl.GL_LINES, None, ("v2f", (y_min, x_min, y_min, x_max)))
        main_batch.add(2, gl.GL_LINES, None, ("v2f", (y_max, x_max, y_min, x_max)))
        main_batch.add(2, gl.GL_LINES, None, ("v2f", (y_max, x_max, y_max, x_min)))
        main_batch.draw()
        # reset color
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def display_text(self, text, x, y, color=(0, 0, 204, 255), anchor_x="left", anchor_y="top"):  # RGBA
        x = int(self.img_height - x)
        y = int(y)
        label = pyglet.text.Label(
            text, font_name="Ariel", color=color, font_size=8, bold=True, x=y, y=x, anchor_x=anchor_x, anchor_y=anchor_y
        )
        label.draw()

    def render(self):
        self.window.flip()

    def save_gif(self, filename=None, arr=None):
        arr[0].save(filename, save_all=True, append_images=arr[1:], duration=500, quality=95)

    def close(self):
        try:
            self.window.close()
        except:
            pass

    def __del__(self):
        self.close()


# TODO: remove tese test lines
import numpy as np
import time

# img = np.random.random(size=(500, 500, 3))*255
img = np.load("testimg.npy")
viewer = SimpleImageViewer(img.shape[0], img.shape[1], scale_y=1, scale_x=1, caption="test image")
viewer.draw_image(img)
viewer.display_text("Hello world", 10, 10)
viewer.render()
time.sleep(1)

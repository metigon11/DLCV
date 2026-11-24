import cv2
import numpy as np

from typing import Tuple
from math import floor, ceil

class DrawWidget(object):
    def __init__(self, window_name: str, canvas_size: Tuple[int, int], display_factor: float, draw_radius: float = 1):
        self.window_name = window_name
        self._original_image = np.zeros(canvas_size, dtype=np.float32)
        self._display_factor = display_factor
        self._draw_radius = draw_radius

        self._MOUSE_LBUTTON = False

        self._clone = self._original_image.copy()
        self._display = self._get_resized()

        self._max_distance_squared = self._draw_radius**2*2

        self._updated = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _get_resized(self):
        return cv2.resize(self._clone, (0, 0), fx=self._display_factor, fy=self._display_factor, interpolation=cv2.INTER_NEAREST)

    def _draw_circle(self, x, y):
        # Resize x and y
        x = x / self._display_factor
        y = y / self._display_factor

        iter_radius = ceil(self._draw_radius)
        # Draw circle
        for i in range(-iter_radius, iter_radius+1):
            for j in range(-iter_radius, iter_radius+1):
                draw_y, draw_x = floor(y+i), floor(x+j)
                if 0 <= draw_y < self._clone.shape[0] and 0 <= draw_x < self._clone.shape[1]:
                    distance_squared = (draw_y+0.5-y)**2+(draw_x+0.5-x)**2
                    factor = max(1 - (distance_squared / self._max_distance_squared)**4, 0) if distance_squared > 0 else 1
                    self._clone[draw_y, draw_x] += factor

        # Display
        self.show_image()

    def _mouse_callback(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._MOUSE_LBUTTON = True
            self._draw_circle(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._MOUSE_LBUTTON = False
        elif event == cv2.EVENT_MOUSEMOVE and self._MOUSE_LBUTTON:
            self._draw_circle(x, y)
        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._clone = self._original_image.copy()
            self.show_image()

    def show_image(self, update:bool = True):
        self._display = self._get_resized()
        cv2.imshow(self.window_name, self._display) 

        self._updated = True

    def get_display_image(self):
        return self._display

    def get_internal_image(self):
        return self._clone
    
    def get_updated_state(self):
        tmp = self._updated
        self._updated = False
        return tmp

if __name__ == '__main__':
    draw_line_widget = DrawWidget("image", (28, 28), 20, 0.75)
    while True:
        cv2.imshow('image', draw_line_widget.get_display_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
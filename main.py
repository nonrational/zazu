#!/usr/bin/env python3

from djitellopy import Tello
import cv2, math, time

print("init...")

me = Tello()
me.connect()
me.streamon()

print(me.get_battery())
frame_read = me.get_frame_read()

me.takeoff()

while True:
    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break
    elif key == ord('w'):
        me.move_forward(30)
    elif key == ord('s'):
        me.move_back(30)
    elif key == ord('a'):
        me.move_left(30)
    elif key == ord('d'):
        me.move_right(30)
    elif key == ord('e'):
        me.rotate_clockwise(30)
    elif key == ord('q'):
        me.rotate_counter_clockwise(30)
    elif key == ord('r'):
        me.move_up(30)
    elif key == ord('f'):
        me.move_down(30)

me.land()

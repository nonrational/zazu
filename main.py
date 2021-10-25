#!/usr/bin/env python3

from djitellopy import Tello
import cv2, math, time, os

FLY=False

# def flight_command(me, command, arg)
#   if FLY:

def move_and_wait(me):
  frame_read = me.get_frame_read()

  while True:
    img = frame_read.frame
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
      return
    # elif key == ord('w'):
    #   me.move_forward(30)
    # elif key == ord('s'):
    #   me.move_back(30)
    # elif key == ord('a'):
    #   me.move_left(30)
    # elif key == ord('d'):
    #   me.move_right(30)
    # elif key == ord('e'):
    #   me.rotate_clockwise(30)
    # elif key == ord('q'):
    #   me.rotate_counter_clockwise(30)
    # elif key == ord('r'):
    #   me.move_up(30)
    # elif key == ord('f'):
    #   me.move_down(30)


print("init...")

me = Tello()

me.connect()

print(me.get_battery())

# # just in case
# me.streamoff()
# me.streamon()
# me.takeoff() if FLY else print("noop")

# move_and_wait(me)

# cv2.destroyAllWindows()

me.initiate_throw_takeoff()
time.sleep(5)


me.end()

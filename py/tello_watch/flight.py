from djitellopy import Tello


class Flight:
    def __init__(self, fly=True):
        self.fly = fly
        self.tello = Tello()

    def connect(self):
        self.tello.connect()

    def battery(self):
        return self.tello.get_battery()

    def state(self):
        try:
            return self.tello.get_current_state()
        except Exception:
            return {}

    def start_video(self):
        self.tello.streamoff()
        self.tello.streamon()

    def get_frame(self):
        return self.tello.get_frame_read().frame

    def takeoff(self):
        if self.fly:
            self.tello.takeoff()

    def send(self, cmd):
        if self.fly:
            self.tello.send_rc_control(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

    def land(self):
        if self.fly:
            self.tello.land()

    def end(self):
        self.tello.end()

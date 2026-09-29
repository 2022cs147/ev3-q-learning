#!/usr/bin/env pybricks-micropython

from pybricks.ev3devices import ColorSensor, Motor
from pybricks.hubs import EV3Brick
from pybricks.media.ev3dev import Font
from pybricks.parameters import Button, Port
from pybricks.robotics import DriveBase
from pybricks.tools import StopWatch, wait

try:
    import ujson as json
except ImportError:
    import json

try:
    import urandom as random
except ImportError:
    import random


# ---- Hardware ----
WHEEL_DIAMETER_MM = 56
AXLE_TRACK_MM = 114

# ---- Files ----
CALIBRATION_FILE = "calibration.json"
QTABLE_FILE = "qtable.json"
SEED_FILE = "qtable-seed.json"      # fallback if qtable.json is missing/bad

# ---- Calibration ----
DEFAULT_CALIBRATION = (4, 22, 40)   # black mat, tape edge, tape middle
CALIBRATION_SAMPLE_MS = 3000
CALIBRATION_MIN_GAP = 3             # edge must sit this far inside black..white

# ---- Motion ----
# Speeds are 60 % of the 2026-09-30 refactor (70-130 / 10 / 70 / 60), so a
# forward step (10-20 mm) can no longer cross the 5 cm tape in one or two steps.
FORWARD_SPEED_MIN = 42              # mm/s, first forward step after a turn
FORWARD_SPEED_MAX = 78              # mm/s, ceiling on a long straight
FORWARD_RAMP = 3                    # mm/s added per consecutive forward step
FORWARD_MS = 250                    # longest forward step
FORWARD_SLICE_MS = 20               # band re-checked this often going forward
FORWARD_MIN_MS = 40                 # ignore band changes before this (noise)
TURN_SPEED = 6                      # mm/s creep while turning
TURN_RATE = 42                      # deg/s
TURN_SLICE_MS = 30                  # band re-checked this often during a turn
# Every corner on the track is 90 deg, so a turn this long without the band
# changing is going the wrong way: the tape-side belief is flipped.
TURN_MAX_DEG = 110
REVERSE_SPEED = 36
REVERSE_MS = 250
LOST_HALT_MM = 250.0                # distance without tape before halting

# ---- Learning ----
ALPHA = 0.1
GAMMA = 0.9
EPSILON_FRESH = 0.40
EPSILON_RESUME = 0.15              # top-up of the seed: ~550 steps to the floor
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998
SAVE_EVERY_STEPS = 25

# ---- Screen ----
SCREEN_PAD = 6
SCREEN_EVERY = 4


FORWARD, LEFT, RIGHT, REVERSE = 0, 1, 2, 3
ACTION_COUNT = 4
ACTION_NAMES = ("FWD", "LEFT", "RIGHT", "REV")

SIDE_RIGHT, SIDE_LEFT = 0, 1
SIDE_NAMES = ("tapeR", "tapeL")

DARK, EDGE, LIGHT = 0, 1, 2
BAND_NAMES = ("DARK", "EDGE", "LIGHT")

STATE_COUNT = 2 * 3

# (band before, turn, band after) -> side the tape is on. Anything not listed
# leaves the belief unchanged.
SIDE_FROM_TRANSITION = {
    (EDGE, RIGHT, LIGHT): SIDE_RIGHT,
    (LIGHT, LEFT, EDGE): SIDE_RIGHT,
    (EDGE, LEFT, DARK): SIDE_RIGHT,
    (DARK, RIGHT, EDGE): SIDE_RIGHT,
    (EDGE, RIGHT, DARK): SIDE_LEFT,
    (DARK, LEFT, EDGE): SIDE_LEFT,
    (EDGE, LEFT, LIGHT): SIDE_LEFT,
    (LIGHT, RIGHT, EDGE): SIDE_LEFT,
}


ev3 = EV3Brick()
font = Font(size=10)
ev3.screen.set_font(font)
sensor = ColorSensor(Port.S1)
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
robot = DriveBase(left_motor, right_motor,
                  wheel_diameter=WHEEL_DIAMETER_MM, axle_track=AXLE_TRACK_MM)


class PauseException(Exception):
    pass


# ---------- Screen and buttons ----------

def show(*lines):
    ev3.screen.clear()
    for i, line in enumerate(lines):
        ev3.screen.draw_text(SCREEN_PAD, SCREEN_PAD + i * (font.height + 2),
                             str(line))


def wait_for_release():
    while ev3.buttons.pressed():
        wait(20)


def wait_press(button):
    while button not in ev3.buttons.pressed():
        wait(20)
    wait_for_release()


def ask(title, up_label, down_label):
    show(title, "", "UP   = " + up_label, "DOWN = " + down_label)
    while True:
        pressed = ev3.buttons.pressed()
        if Button.UP in pressed or Button.DOWN in pressed:
            wait_for_release()
            return Button.UP in pressed
        wait(20)


def check_pause():
    if Button.LEFT in ev3.buttons.pressed():
        halt()
        raise PauseException()


def interruptible_wait(ms):
    watch = StopWatch()
    while watch.time() < ms:
        check_pause()
        wait(min(15, ms - watch.time()))


# ---------- Calibration ----------

calibration = DEFAULT_CALIBRATION
dark_cut = 0.0
light_cut = 100.0


def apply_calibration(black, edge, white):
    global calibration, dark_cut, light_cut
    calibration = (black, edge, white)
    dark_cut = (black + edge) / 2
    light_cut = (edge + white) / 2


def load_calibration():
    try:
        with open(CALIBRATION_FILE) as handle:
            saved = json.load(handle)
        black, white = saved["black"], saved["white"]
        apply_calibration(black, saved.get("edge", (black + white) / 2), white)
    except Exception:
        apply_calibration(*DEFAULT_CALIBRATION)


def sample_surface(name, hint):
    show("Calibrate " + name, "", hint, "", "CENTER = start",
         "(samples for %d s)" % (CALIBRATION_SAMPLE_MS // 1000))
    wait_press(Button.CENTER)
    ev3.speaker.beep(1000, 100)

    readings = []
    watch = StopWatch()
    while watch.time() < CALIBRATION_SAMPLE_MS:
        readings.append(sensor.reflection())
        wait(20)
    ev3.speaker.beep(1500, 100)

    average = sum(readings) / len(readings)
    show(name, "", "avg %.1f" % average,
         "min %d  max %d" % (min(readings), max(readings)))
    wait(1500)
    return average


def run_calibration():
    black = sample_surface("BLACK", "drag over the black mat")
    edge = sample_surface("EDGE", "drag ALONG the tape edge")
    white = sample_surface("WHITE", "drag along tape middle")

    if not black + CALIBRATION_MIN_GAP < edge < white - CALIBRATION_MIN_GAP:
        show("BAD CALIBRATION", "", "b %.0f e %.0f w %.0f" % (black, edge, white),
             "keeping old values")
        wait(2500)
        return

    apply_calibration(black, edge, white)
    try:
        with open(CALIBRATION_FILE, "w") as handle:
            json.dump({"black": round(black, 1), "edge": round(edge, 1),
                       "white": round(white, 1)}, handle)
    except Exception:
        pass

    show("Calibrated", "", "b %.0f e %.0f w %.0f" % (black, edge, white),
         "DARK  < %.1f" % dark_cut, "LIGHT > %.1f" % light_cut)
    wait(2500)


# ---------- Sensing ----------

def read_reflection():
    """Median of three reads."""
    a, b, c = sensor.reflection(), sensor.reflection(), sensor.reflection()
    return a + b + c - min(a, b, c) - max(a, b, c)


def band_of(reflection):
    if reflection < dark_cut:
        return DARK
    if reflection > light_cut:
        return LIGHT
    return EDGE


# ---------- Q-table ----------

def blank_table():
    return [[0.0] * ACTION_COUNT for _ in range(STATE_COUNT)]


def load_qtable():
    for path in (QTABLE_FILE, SEED_FILE):
        try:
            with open(path) as handle:
                table = json.load(handle)
            if (len(table) == STATE_COUNT
                    and all(len(row) == ACTION_COUNT for row in table)):
                return [[float(v) for v in row] for row in table]
        except Exception:
            pass
    return blank_table()


def save_qtable(table):
    try:
        with open(QTABLE_FILE, "w") as handle:
            json.dump(table, handle)
    except Exception:
        pass


def has_values(table):
    return any(abs(v) > 0.01 for row in table for v in row)


def random_float():
    try:
        return random.getrandbits(16) / 65535.0
    except AttributeError:
        return random.random()


def random_int(limit):
    return random.getrandbits(8) % limit if limit > 1 else 0


def choose_action(row, epsilon):
    if epsilon > 0 and random_float() < epsilon:
        return random_int(ACTION_COUNT)
    best = max(row)
    tied = [a for a in range(ACTION_COUNT) if row[a] >= best]
    return tied[random_int(len(tied))]


# ---------- Motion ----------

forward_speed = FORWARD_SPEED_MIN


def halt():
    robot.stop()
    left_motor.brake()
    right_motor.brake()


def do_forward(band):
    """Drive until the band changes, at most FORWARD_MS."""
    global forward_speed
    forward_speed = min(forward_speed + FORWARD_RAMP, FORWARD_SPEED_MAX)
    robot.drive(forward_speed, 0)
    watch = StopWatch()
    while watch.time() < FORWARD_MS:
        interruptible_wait(FORWARD_SLICE_MS)
        if (watch.time() >= FORWARD_MIN_MS
                and band_of(read_reflection()) != band):
            return


def do_turn(action, band):
    """Turn until the band changes. True if it gave up at TURN_MAX_DEG."""
    global forward_speed
    forward_speed = FORWARD_SPEED_MIN
    start = robot.angle()
    robot.drive(TURN_SPEED, TURN_RATE if action == RIGHT else -TURN_RATE)
    while True:
        interruptible_wait(TURN_SLICE_MS)
        if band_of(read_reflection()) != band:
            return False
        if abs(robot.angle() - start) >= TURN_MAX_DEG:
            return True


def do_reverse():
    global forward_speed
    forward_speed = FORWARD_SPEED_MIN
    robot.drive(-REVERSE_SPEED, 0)
    interruptible_wait(REVERSE_MS)


def apply_action(action, band):
    """True if a turn gave up without the band changing."""
    if action == FORWARD:
        do_forward(band)
    elif action == REVERSE:
        do_reverse()
    else:
        return do_turn(action, band)
    return False


def probe_side(side):
    """One right turn to find out which side the tape is on."""
    show("probing side")
    band = band_of(read_reflection())
    do_turn(RIGHT, band)
    new_band = band_of(read_reflection())
    halt()
    side = SIDE_FROM_TRANSITION.get((band, RIGHT, new_band), side)
    show("tape: " + SIDE_NAMES[side])
    wait(700)
    return side


def resync():
    reflection = read_reflection()
    return reflection, band_of(reflection), 0.0, robot.distance()


# ---------- Startup ----------

load_calibration()
if ask("CALIBRATE?", "calibrate", "use saved"):
    run_calibration()

training = ask("MODE", "TRAIN", "RUN")
qtable = load_qtable()

epsilon = 0.0
if training:
    if has_values(qtable) and not ask("TABLE FOUND", "continue", "WIPE"):
        qtable = blank_table()
        ev3.speaker.beep(400, 300)
    epsilon = EPSILON_RESUME if has_values(qtable) else EPSILON_FRESH

show("TRAIN" if training else "RUN", "eps %d" % int(epsilon * 100), "",
     "Place on the EDGE", "CENTER to start")
wait_press(Button.CENTER)
ev3.speaker.beep(1000, 150)

side = SIDE_RIGHT
try:
    side = probe_side(side)
except PauseException:
    wait_for_release()


# ---------- Main loop ----------

steps = 0
visits = [0] * STATE_COUNT
reflection, band, dark_mm, last_mm = resync()

try:
    while Button.CENTER not in ev3.buttons.pressed():
        try:
            check_pause()
            state = side * 3 + band
            action = choose_action(qtable[state], epsilon)

            if steps % SCREEN_EVERY == 0:
                show("%s  %s" % (SIDE_NAMES[side], BAND_NAMES[band]),
                     "%s  refl %d" % (ACTION_NAMES[action], reflection),
                     "eps %d  step %d" % (int(epsilon * 100), steps),
                     "dark %d mm" % int(dark_mm))

            gave_up = apply_action(action, band)

            reflection = read_reflection()
            new_band = band_of(reflection)
            old_side = side
            if gave_up:
                side = 1 - side
            else:
                side = SIDE_FROM_TRANSITION.get((band, action, new_band), side)
            if side != old_side:
                print("step", steps, "side", SIDE_NAMES[old_side], "->",
                      SIDE_NAMES[side], "after", BAND_NAMES[band],
                      ACTION_NAMES[action], BAND_NAMES[new_band],
                      "(gave up)" if gave_up else "")
            now_mm = robot.distance()
            dark_mm = dark_mm + abs(now_mm - last_mm) if new_band == DARK else 0.0
            last_mm = now_mm
            band = new_band

            if training:
                reward = 10.0 if band == EDGE else -10.0
                row = qtable[state]
                target = reward + GAMMA * max(qtable[side * 3 + band])
                row[action] += ALPHA * (target - row[action])
                visits[state] += 1
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            steps += 1
            if training and steps % SAVE_EVERY_STEPS == 0:
                save_qtable(qtable)

            if dark_mm >= LOST_HALT_MM:
                halt()
                ev3.speaker.beep(300, 400)
                show("LOST - halted", "", "Put it back on", "the edge, then",
                     "press CENTER")
                wait_press(Button.CENTER)
                ev3.speaker.beep(1000, 150)
                side = probe_side(side)
                reflection, band, dark_mm, last_mm = resync()

        except PauseException:
            wait_for_release()
            ev3.speaker.beep(400, 150)
            show("PAUSED", "", "CENTER = resume")
            wait_press(Button.CENTER)
            ev3.speaker.beep(1000, 150)
            reflection, band, dark_mm, last_mm = resync()

finally:
    halt()
    if training:
        save_qtable(qtable)
    show("STOPPED", "", "steps: %d" % steps)

    print("calibration black/edge/white", calibration,
          "DARK <", dark_cut, "LIGHT >", light_cut)
    print("Q-table after", steps, "steps:")
    for index, row in enumerate(qtable):
        best = row.index(max(row))
        print(index, SIDE_NAMES[index // 3], BAND_NAMES[index % 3],
              ["%.1f" % v for v in row], "best", ACTION_NAMES[best],
              "n", visits[index])

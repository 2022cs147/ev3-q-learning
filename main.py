#!/usr/bin/env pybricks-micropython
"""Edge-following robot that LEARNS forward / reverse / left / right by Q
learning. Obstacle avoidance and re-finding the path after a detour are
scripted, which the brief explicitly allows.

THE IDEA
--------
One downward colour sensor returns an UNSIGNED number. A single reading
therefore cannot say whether the dark is on the left or the right. That is the
one fact a reading cannot carry and the state has to, so the state is exactly
two things:

  tape_side  -- which side the tape is on. NOT sensed from one reading; INFERRED
                from the BAND TRANSITION a turn produced. Turn right, cross into
                the light => the tape is to the right. Eight rules, pure
                geometry, in SIDE_FROM_TRANSITION. They map (what I did, what I
                then crossed) onto (where the tape is). They never map onto (what
                to do next) -- that stays in the table.

  band       -- DARK / EDGE / LIGHT, straight from the reflection reading.

state = tape_side * 3 + band  ->  6 states x 4 actions.

ALIGNED WITH THE WORKING REFERENCE IMPLEMENTATION
-------------------------------------------------
qtable.json holds an operator-supplied table that was trained by a different
program. Loading those numbers into a program whose actions MEAN something else
gets you the numbers without the behaviour, so four things were brought into
line with the program that produced them. Each is load-bearing:

1. A TURN RUNS UNTIL THE BAND CHANGES.
   Previously a turn was a fixed 45 ms, 2 degree nudge and the table was
   re-consulted after each one, so a 90 degree corner needed about forty-four
   consecutive correct decisions and any one weak row stalled it. Now a turn is a
   macro-action: it sweeps at TURN_RATE until band_of(reflection) differs from
   the band the decision was taken in, then stops. Self-terminating, so it cannot
   under-rotate at a sharp corner nor over-rotate on a straight. See do_turn.

   This also solves the 5 cm tape plateau. The tape body is far wider than the
   sensor spot, so LIGHT carries no gradient -- nothing in the reading says which
   way its own edge lies. The old loop crossed that plateau in a dozen-odd blind
   fixed steps. One macro-turn crosses it in a single action that ends, by
   construction, on the far boundary.

2. FORWARD IS ONE CONSTANT SPEED.
   Every forward step drives at FORWARD_SPEED, close to the turn speed, so there
   is no speed jump between turning and driving straight. That is where the
   smoothness comes from, and it is geometry rather than anything in the Q-table.

3. THE SIDE BELIEF MOVES ONLY ON A REAL BAND CROSSING.
   The old rule flipped on the SIGN of any reflection change larger than 3. On a
   wide tape the reading wanders several units with no lateral meaning, so the
   belief flipped on that wander. Requiring a band crossing means the belief
   moves only when something genuinely lateral happened.

4. THE REWARD IS +10 ON THE EDGE, -10 OFF IT, AND NOTHING ELSE.
   The previous version had a progress gate, a per-step cost for being lost, a
   re-acquisition bonus and an error gradient. That is a different objective, so
   a TRAIN run would have dragged every row toward a policy the supplied numbers
   were not describing. The simple reward is demonstrably the right one: the
   supplied Q(tapeR, DARK, RIGHT) = 38.34 is within 0.05 of
   10 + GAMMA * max(tapeR EDGE row) = 10 + 0.9 * 31.44 = 38.29. Same reward in,
   same table out.

A startup PROBE TURN now establishes tape_side before the first decision, as the
reference does, instead of opening on a guess. It is re-run after an obstacle
detour and after a hand replacement, because both invalidate the belief.

WHAT FOLLOWS FROM THE SUPPLIED TABLE
------------------------------------
  * A corner and a genuine departure are the same state. Both DARK rows prefer a
    turn, so the robot turns at both -- good for the 30 marks of turning and
    smoothness, useless when it has actually left the tape. LOST_HALT_MM is what
    stops it leaving the table; dark_mm is that counter and nothing else.
  * REVERSE is in the action set and is learnable on equal terms -- the reward
    gives it no surcharge and no discount -- but the supplied table scores it 0.0
    in all six rows against turns at 35-38, so it never wins greedily. In RUN
    mode (epsilon 0) the robot will not reverse at all. Only a TRAIN run can
    raise it, and the 20-mark reverse item is not demonstrable from the seed.

SEEDED VALUES
-------------
qtable.json is SEEDED from the operator-supplied table -- see seed_qtable.py,
which holds the 18 numbers and the mapping. Turn polarity is therefore written
into the program rather than learned. This is a deliberate operator decision,
recorded here so nobody has to reverse-engineer it from the JSON.
"""

from pybricks.ev3devices import ColorSensor, InfraredSensor, Motor
from pybricks.hubs import EV3Brick
from pybricks.media.ev3dev import Font, SoundFile
from pybricks.parameters import Button, Direction, Port
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


# ============================================================
# Hardware -- measured, see ../HARDWARE.md
# ============================================================

LEFT_MOTOR_PORT = Port.B
RIGHT_MOTOR_PORT = Port.C
COLOR_SENSOR_PORT = Port.S1
IR_SENSOR_PORT = Port.S4

LEFT_MOTOR_INVERTED = False
RIGHT_MOTOR_INVERTED = False

WHEEL_DIAMETER_MM = 56
AXLE_TRACK_MM = 114


# ============================================================
# Tuning
# ============================================================

CALIBRATION_FILE = "calibration.json"
QTABLE_FILE = "qtable.json"
# Pristine copy of the seeded policy, written by seed_qtable.py and never
# written by the robot. Used only when qtable.json is missing or is the wrong
# shape -- which it will be on any brick still carrying the 10-row table from
# the previous state space. Without this, that stale file is rejected and the
# robot silently starts blank on demo day.
SEED_FILE = "qtable-seed.json"

# ---- Calibration --------------------------------------------------------
# Three surfaces are sampled: the black mat, the tape EDGE itself, and the tape
# middle. DARK ends halfway between black and edge, LIGHT starts halfway between
# edge and white, so the EDGE band is centred on a measured edge reading rather
# than on a fraction of the black/white span.
DEFAULT_CALIBRATION = (4, 22, 40)   # black mat, tape edge, tape middle
CALIBRATION_SAMPLE_MS = 3000
CALIBRATION_MIN_GAP = 3             # edge must sit this far inside black..white

# ---- Screen -------------------------------------------------------------
SCREEN_PAD = 6

# ---- Forward ------------------------------------------------------------
# One constant speed. Kept low so going from a turn into a straight does not
# jerk -- geometry, not the Q-table.
FORWARD_SPEED = 55         # mm/s
FORWARD_MS = 250           # max duration of one forward step

# ---- Turns are CLOSED-LOOP MACRO-ACTIONS --------------------------------
# A turn runs UNTIL THE BAND CHANGES, not for a fixed time. It is re-issued in
# TURN_SLICE_MS slices and stops the moment band_of(reflection) differs from the
# band the decision was taken in.
#
# This is the single most important difference from the previous version, and it
# is the semantics the table in qtable.json was learned under. A fixed 45 ms /
# 2 degree kick makes "turn" mean "nudge, then re-decide", so getting round a
# 90 degree corner needs ~44 consecutive correct decisions and any one weak row
# stalls it. Turn-until-the-band-changes makes "turn" mean "sweep across the
# boundary": self-terminating, so it cannot under-rotate at a sharp corner nor
# over-rotate on a straight. It is also what makes the side belief trustworthy,
# because every turn now ends on a real band transition rather than on noise.
TURN_SPEED = 10            # mm/s of creep while turning -- nearly a pivot
TURN_RATE = 60             # deg/s -- slow enough not to jump the thin EDGE band
TURN_SLICE_MS = 20         # how often the band is re-checked during a turn
TURN_MAX_DEG = 200         # safety: give up rather than spin forever

REVERSE_SPEED = 85         # mm/s backward
REVERSE_MS = 250           # one reverse step, same duration as a forward

# Used only by the scripted straights (obstacle detour, re-finding sweep).
DRIVE_SPEED = 70

SCREEN_EVERY = 4           # Steps are long now, so redrawing is affordable.

ALPHA = 0.1                # As the reference used
GAMMA = 0.9

EPSILON_FRESH = 0.40       # Blank table
EPSILON_RESUME = 0.10      # Topping up a trained table
EPSILON_MIN = 0.05
# A step is now a whole macro-action -- 250 ms of forward, or however long the
# boundary takes to cross -- so a lap is a few hundred steps, not several
# thousand. 0.9995 per step would never reach the floor inside a demo. 0.995
# takes 0.40 -> 0.05 in about 415 steps, roughly one lap of the big square.
EPSILON_DECAY = 0.995
SAVE_EVERY_STEPS = 25

# Safety only: total path covered without seeing the tape before the program
# stops and waits for a human. A turn barely translates, so a long corner sweep
# cannot trip this; driving away in a straight line can. It does NOT rescue
# itself.
LOST_HALT_MM = 250.0

# Obstacle handling. InfraredSensor.distance() is a RELATIVE 0-100 value,
# NOT centimetres -- do not copy a threshold from an ultrasonic example.
IR_TRIGGER = 22
IR_CONFIRM_READINGS = 2
DETOUR_BACK_MM = 60
DETOUR_TURN_DEG = 85
DETOUR_SIDE_MM = 180
DETOUR_PAST_MM = 240

# Scripted sweep used ONLY to re-find the path after an obstacle detour.
SWEEP_STEP_DEG = 12
SWEEP_MAX_ARCS = 8
SWEEP_CREEP_MM = 35
SETTLE_STEP_DEG = 5
SETTLE_MAX_NUDGES = 8

TRACE_LIMIT = 4000


# ============================================================


ev3 = EV3Brick()
font = Font(size=10)
ev3.screen.set_font(font)

left_motor = Motor(
    LEFT_MOTOR_PORT,
    positive_direction=(Direction.COUNTERCLOCKWISE if LEFT_MOTOR_INVERTED
                        else Direction.CLOCKWISE),
)
right_motor = Motor(
    RIGHT_MOTOR_PORT,
    positive_direction=(Direction.COUNTERCLOCKWISE if RIGHT_MOTOR_INVERTED
                        else Direction.CLOCKWISE),
)
line_sensor = ColorSensor(COLOR_SENSOR_PORT)
ir_sensor = InfraredSensor(IR_SENSOR_PORT)

robot = DriveBase(left_motor, right_motor,
                  wheel_diameter=WHEEL_DIAMETER_MM, axle_track=AXLE_TRACK_MM)


# ---------- State space ----------

FORWARD, LEFT, RIGHT, REVERSE = 0, 1, 2, 3
ACTION_COUNT = 4
ACTION_NAMES = ("FWD", "LEFT", "RIGHT", "REV")

SIDE_RIGHT, SIDE_LEFT = 0, 1          # Which side the tape is on
SIDE_NAMES = ("tapeR", "tapeL")

# The three reflection bands ARE the situation now. The dark-age split that
# used to sit between them is gone; see the header.
DARK, EDGE, LIGHT = 0, 1, 2
BAND_COUNT = 3
BAND_NAMES = ("DARK", "EDGE", "LIGHT")

STATE_COUNT = 2 * BAND_COUNT           # 6


# ---------- Calibration ----------

calibration = DEFAULT_CALIBRATION
DARK_CUT = 0.0
LIGHT_CUT = 100.0
EDGE_TARGET = 50.0
EDGE_HALF = 50.0


def apply_calibration(black, edge, white):
    """Install a black/edge/white triple and rebuild the band cuts from it."""
    global calibration, DARK_CUT, LIGHT_CUT, EDGE_TARGET, EDGE_HALF

    calibration = (black, edge, white)
    DARK_CUT = (black + edge) / 2
    LIGHT_CUT = (edge + white) / 2
    EDGE_TARGET = (DARK_CUT + LIGHT_CUT) / 2.0
    EDGE_HALF = (LIGHT_CUT - DARK_CUT) / 2.0


def load_calibration():
    try:
        with open(CALIBRATION_FILE) as handle:
            saved = json.load(handle)
        black, white = saved["black"], saved["white"]
        apply_calibration(black, saved.get("edge", (black + white) / 2), white)
    except Exception:
        apply_calibration(*DEFAULT_CALIBRATION)


# ---------- Small helpers ----------

def random_float():
    """Uniform float in [0, 1). MicroPython's urandom has no random()."""
    try:
        return random.getrandbits(16) / 65535.0
    except AttributeError:
        return random.random()


def random_int(limit):
    if limit <= 1:
        return 0
    return random.getrandbits(8) % limit


def read_reflection():
    """Median of three reads. Costs ~3 ms and removes single-sample spikes."""
    a = line_sensor.reflection()
    b = line_sensor.reflection()
    c = line_sensor.reflection()
    return a + b + c - min(a, b, c) - max(a, b, c)


def band_of(reflection):
    """DARK / EDGE / LIGHT, the reference implementation's three-way split."""
    if reflection < DARK_CUT:
        return DARK
    if reflection > LIGHT_CUT:
        return LIGHT
    return EDGE


def tape_visible(reflection):
    """True when the sensor can see tape at all -- the edge or the body."""
    return reflection >= DARK_CUT


def halt():
    robot.stop()
    left_motor.brake()
    right_motor.brake()


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


# ---------- Q table ----------

def blank_table():
    return [[0.0] * ACTION_COUNT for _ in range(STATE_COUNT)]


def read_table(path):
    """Return a correctly shaped table from path, or None."""
    try:
        with open(path, "r") as handle:
            table = json.load(handle)
        if (len(table) == STATE_COUNT
                and all(len(row) == ACTION_COUNT for row in table)):
            return [[float(v) for v in row] for row in table]
    except Exception:
        pass
    return None


def load_qtable():
    """Working table first; fall back to the seed; blank as a last resort."""
    table = read_table(QTABLE_FILE)
    if table is not None:
        return table, True

    table = read_table(SEED_FILE)
    if table is not None:
        return table, True

    return blank_table(), False


def save_qtable(table):
    try:
        with open(QTABLE_FILE, "w") as handle:
            json.dump(table, handle)
    except Exception:
        pass


def has_learned_values(table):
    return any(any(abs(v) > 0.01 for v in row) for row in table)


def choose_action(table, state, epsilon):
    if epsilon > 0.0 and random_float() < epsilon:
        return random_int(ACTION_COUNT)

    # Ties are broken at RANDOM, not by index order. With index order a blank
    # row always returns action 0 -- FORWARD -- so an untrained robot sitting in
    # the black drove itself steadily further out (70 % forward against 10 %
    # reverse at EPSILON_FRESH, a net 1.9 mm per step AWAY from the tape) and
    # the reverse drill could never close the distance. Random tie-breaking
    # makes an untrained row genuinely undecided, which is what a blank row
    # means. It is also not a hint: all four actions are treated alike.
    row = table[state]
    best = row[0]
    for index in range(1, ACTION_COUNT):
        if row[index] > best:
            best = row[index]

    tied = [index for index in range(ACTION_COUNT) if row[index] >= best]
    if len(tied) == 1:
        return tied[0]
    return tied[random_int(len(tied))]


# ---------- The two inferred facts ----------

# Which side is the tape on? Read off the BAND TRANSITION a turn produced.
#
#   tape on the RIGHT  <=>  turning right moves you toward the tape
#                           (DARK->EDGE, EDGE->LIGHT) and turning left moves you
#                           away from it (EDGE->DARK, LIGHT->EDGE)
#   tape on the LEFT   <=>  exactly the mirror
#
# These are the reference implementation's m_x and m_y sets, unchanged, written
# as one mapping. Anything NOT listed -- a turn that crossed no boundary, a
# forward, a reverse, a straight DARK->LIGHT jump -- leaves the belief ALONE.
#
# That is the real change from the old rule, which flipped on the SIGN of any
# reflection change larger than 3. On a 5 cm tape the sensor spends a dozen
# steps on the body, where the reading wanders several units with no lateral
# meaning at all, and the old rule flipped the belief on that wander. Requiring
# an actual band crossing means the belief moves only when something genuinely
# lateral happened.
#
# The belief is also what lets ONE table drive the contour in both directions:
# flipped, it follows the same edge the other way round. That is the 20-mark
# clockwise-and-anticlockwise item, with no second table.
SIDE_FROM_TRANSITION = {
    (EDGE,  RIGHT, LIGHT): SIDE_RIGHT,
    (LIGHT, LEFT,  EDGE):  SIDE_RIGHT,
    (EDGE,  LEFT,  DARK):  SIDE_RIGHT,
    (DARK,  RIGHT, EDGE):  SIDE_RIGHT,

    (EDGE,  RIGHT, DARK):  SIDE_LEFT,
    (DARK,  LEFT,  EDGE):  SIDE_LEFT,
    (EDGE,  LEFT,  LIGHT): SIDE_LEFT,
    (LIGHT, RIGHT, EDGE):  SIDE_LEFT,
}


def infer_tape_side(side, action, old_band, new_band):
    """Pure geometry: says where the tape IS, never what to do about it."""
    return SIDE_FROM_TRANSITION.get((old_band, action, new_band), side)


def state_index(side, band):
    return side * BAND_COUNT + band


# ---------- Reward ----------

def reward_for(band):
    """+10 on the edge, -10 off it. That is the entire reward.

    This is the reference implementation's reward, and it is the one that
    produced the numbers now in qtable.json. The previous multi-term version --
    a progress gate, a per-step cost for being lost, a re-acquisition bonus and
    an error gradient -- encoded a DIFFERENT objective, so a TRAIN run on top of
    the supplied table would have dragged every row toward a policy those
    numbers were not describing. Same reward in, same table out.

    Nothing here says which way to turn for a given reading, and there is no
    reverse surcharge or discount: REVERSE is priced exactly like the others, by
    whether it lands the sensor back on the edge.
    """
    return 10.0 if band == EDGE else -10.0


# ---------- Motion ----------

def do_forward():
    """Constant-speed straight.

    Ends early the moment the sensor leaves the EDGE band, so a drift is
    corrected after a few millimetres instead of a whole blind 250 ms step.
    """
    robot.drive(FORWARD_SPEED, 0)
    watch = StopWatch()
    watch.reset()
    while watch.time() < FORWARD_MS:
        check_pause()
        wait(15)
        if band_of(read_reflection()) != EDGE:
            return


def do_turn(action, band):
    """Turn until the band changes. Self-terminating -- see TURN_RATE above.

    Never calls halt(): motion stays continuous into whatever comes next, which
    is where the smoothness marks live.
    """
    rate = TURN_RATE if action == RIGHT else -TURN_RATE
    start = robot.angle()
    robot.drive(TURN_SPEED, rate)

    while True:
        interruptible_wait(TURN_SLICE_MS)
        if band_of(read_reflection()) != band:
            return
        if abs(robot.angle() - start) >= TURN_MAX_DEG:
            return


def do_reverse():
    robot.drive(-REVERSE_SPEED, 0)
    interruptible_wait(REVERSE_MS)


def apply_action(action, band):
    """Run one action to completion.

    Unlike the old fixed-LOOP_MS version, a step's duration is decided by the
    action: 250 ms for a straight or a reverse, and as long as the boundary takes
    for a turn.
    """
    if action == FORWARD:
        do_forward()
    elif action == REVERSE:
        do_reverse()
    else:
        do_turn(action, band)


def probe_tape_side(side):
    """One right turn to find out which side the tape is on.

    The reference implementation does exactly this before its run loop, and it
    matters: tape_side cannot be read from a single reflection, so without a
    probe the opening steps are driven on a guess. A right turn that reaches
    LIGHTER tape says the tape is to the right; one that reaches DARKER says it
    is to the left. A turn that crosses no boundary leaves the guess standing.
    """
    band = band_of(read_reflection())
    do_turn(RIGHT, band)
    new_band = band_of(read_reflection())
    halt()
    return infer_tape_side(side, RIGHT, band, new_band)


class PauseException(Exception):
    pass


def check_pause():
    if Button.LEFT in ev3.buttons.pressed():
        halt()
        raise PauseException()


def interruptible_wait(ms):
    watch = StopWatch()
    watch.reset()
    while watch.time() < ms:
        check_pause()
        wait(min(15, ms - watch.time()))


def interruptible_straight(distance_mm):
    if abs(distance_mm) < 1:
        return
    speed = DRIVE_SPEED if distance_mm > 0 else -DRIVE_SPEED
    start = robot.distance()
    robot.drive(speed, 0)
    try:
        while abs(robot.distance() - start) < abs(distance_mm):
            check_pause()
            wait(10)
    finally:
        halt()


def interruptible_turn(angle_deg, sound=None):
    """Spin in place by angle_deg. If sound is given it plays DURING the turn:
    drive() is non-blocking, so the motors keep turning while play_file blocks,
    and the angle loop below picks up wherever the turn got to."""
    if abs(angle_deg) < 1:
        return
    rate = TURN_RATE if angle_deg > 0 else -TURN_RATE
    start = robot.angle()
    robot.drive(0, rate)
    try:
        if sound is not None:
            ev3.speaker.play_file(sound)
        while abs(robot.angle() - start) < abs(angle_deg):
            check_pause()
            wait(10)
    finally:
        halt()


# ---------- Scripted: obstacle detour and re-finding the path ----------
#
# The brief allows both of these to be scripted, and gives 15 marks for them.
# Nothing here writes a Q update -- a detour is not something an action caused,
# so crediting whatever action happened to be running would teach the robot that
# driving into things pays.

def obstacle_ahead():
    for _ in range(IR_CONFIRM_READINGS):
        if ir_sensor.distance() >= IR_TRIGGER:
            return False
        wait(12)
    return True


def settle_on_edge():
    """Tape is in view -- creep sideways in widening nudges to find its edge."""
    for nudge in range(SETTLE_MAX_NUDGES):
        if band_of(read_reflection()) == EDGE:
            return True
        step = SETTLE_STEP_DEG * (nudge + 1)
        interruptible_turn(step if nudge % 2 == 0 else -step)
    return band_of(read_reflection()) == EDGE


def scan_turn(total_degrees):
    step = SWEEP_STEP_DEG if total_degrees > 0 else -SWEEP_STEP_DEG
    remaining = abs(total_degrees)
    while remaining > 0:
        interruptible_turn(step if remaining >= SWEEP_STEP_DEG else
                           (remaining if step > 0 else -remaining))
        remaining -= SWEEP_STEP_DEG
        if tape_visible(read_reflection()):
            return True
    return False


def find_line():
    """Widening alternating sweep. Used ONLY after an obstacle detour."""
    if band_of(read_reflection()) == EDGE:
        return True

    for arc in range(1, SWEEP_MAX_ARCS + 1):
        heading = SWEEP_STEP_DEG * 2 * arc
        if arc % 2 == 0:
            heading = -heading

        if scan_turn(heading):
            return settle_on_edge()
        if scan_turn(-heading):
            return settle_on_edge()

        interruptible_straight(SWEEP_CREEP_MM)
        if tape_visible(read_reflection()):
            return settle_on_edge()

    return False


def avoid_obstacle(side):
    """Turn around AWAY from the tape, then resume the main loop.

    Positive angles turn right. With the tape on the right we spin left, so the
    sensor sweeps over open floor instead of dragging across the tape body, and
    vice versa. A 180 puts the tape on the opposite side, so the flipped side
    is returned as the new belief.
    """
    halt()
    interruptible_turn(-180 if side == SIDE_RIGHT else 180,
                       sound=SoundFile.ELEPHANT_CALL)
    return SIDE_LEFT if side == SIDE_RIGHT else SIDE_RIGHT


# ---------- Calibration routine ----------

def sample_surface(name, hint):
    show("Calibrate " + name, "", hint, "", "CENTER = start",
         "(samples for %d s)" % (CALIBRATION_SAMPLE_MS // 1000))
    wait_press(Button.CENTER)
    ev3.speaker.beep(1000, 100)

    readings = []
    watch = StopWatch()
    while watch.time() < CALIBRATION_SAMPLE_MS:
        readings.append(line_sensor.reflection())
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
         "DARK  < %.1f" % DARK_CUT, "LIGHT > %.1f" % LIGHT_CUT)
    wait(2500)


def ask(title, up_label, down_label):
    show(title, "", "UP   = " + up_label, "DOWN = " + down_label)
    while True:
        pressed = ev3.buttons.pressed()
        if Button.UP in pressed or Button.DOWN in pressed:
            wait_for_release()
            return Button.UP in pressed
        wait(20)


# ============================================================
# Startup
# ============================================================

load_calibration()

if ask("CALIBRATE?", "calibrate", "use saved"):
    run_calibration()

training = ask("MODE", "TRAIN", "RUN")

qtable, loaded = load_qtable()

if training:
    if loaded and has_learned_values(qtable):
        if not ask("TABLE FOUND", "continue", "WIPE"):
            qtable = blank_table()
            ev3.speaker.beep(400, 300)
    epsilon = (EPSILON_RESUME if has_learned_values(qtable)
               else EPSILON_FRESH)
else:
    epsilon = 0.0

show("TRAIN" if training else "RUN", "eps %d" % int(epsilon * 100), "",
     "Place on the EDGE", "CENTER to start")
wait_press(Button.CENTER)
ev3.speaker.beep(1000, 150)


# ============================================================
# Main loop
# ============================================================

steps = 0
visits = [0] * STATE_COUNT
dark_mm = 0.0                  # Travelled-blind counter. NOT part of the state.
tape_side = SIDE_RIGHT

trace = []

# One probe turn before the first decision, exactly as the reference does it.
# SIDE_RIGHT above is only where the probe starts from: tape_side cannot be read
# out of a single reflection, so without this the opening steps are driven on a
# coin flip, and a wrong side belief means every turn is the wrong way round
# until some later crossing happens to correct it.
try:
    show("probing side")
    tape_side = probe_tape_side(tape_side)
    show("tape: " + SIDE_NAMES[tape_side])
    wait(700)
except PauseException:
    halt()
    wait_for_release()

prev_distance_mm = robot.distance()
reflection = read_reflection()
band = band_of(reflection)
state = state_index(tape_side, band)


def resync():
    """Re-read the world after something moved the robot for us.

    A scripted detour or a hand placement is not a (state, action, next state)
    sample, so it must never reach a Q update.
    """
    global dark_mm, reflection, band, state
    global prev_distance_mm

    dark_mm = 0.0
    prev_distance_mm = robot.distance()
    reflection = read_reflection()
    band = band_of(reflection)
    state = state_index(tape_side, band)


try:
    while True:
        try:
            check_pause()

            if Button.CENTER in ev3.buttons.pressed():
                break

            # Obstacle handling sits entirely outside the RL loop.
            if obstacle_ahead():
                tape_side = avoid_obstacle(tape_side)
                # The 180 flipped which side the tape is on, and avoid_obstacle
                # returns that flipped belief. Still confirm it with a probe --
                # the flip is the prior, the probe answers the question.
                tape_side = probe_tape_side(tape_side)
                resync()
                continue

            action = choose_action(qtable, state, epsilon)

            # Drawn BEFORE the action is issued, so these milliseconds are spent
            # coasting on the PREVIOUS drive command rather than stolen from this
            # step. The old loop drew after issuing, because a step was 45 ms and
            # had to start at once; a macro-action step is 250 ms or more.
            if steps % SCREEN_EVERY == 0:
                show("%s  %s" % (SIDE_NAMES[tape_side], BAND_NAMES[band]),
                     "%s  refl %d" % (ACTION_NAMES[action], reflection),
                     "eps %d  step %d" % (int(epsilon * 100), steps),
                     "dark %d mm" % int(dark_mm))

            # Blocks for the action's OWN duration: 250 ms for a straight or a
            # reverse, and for a turn however long the boundary takes to cross.
            # There is no fixed LOOP_MS any more, which is why the step budget
            # and EPSILON_DECAY were both re-sized.
            apply_action(action, band)

            new_reflection = read_reflection()
            new_band = band_of(new_reflection)

            distance_now = robot.distance()
            progress_mm = distance_now - prev_distance_mm
            prev_distance_mm = distance_now

            # The side belief moves only on a real band crossing produced by a
            # turn -- BANDS in, not raw reflections. See SIDE_FROM_TRANSITION.
            tape_side = infer_tape_side(tape_side, action, band, new_band)

            if tape_visible(new_reflection):
                dark_mm = 0.0
            else:
                dark_mm += abs(progress_mm)

            new_state = state_index(tape_side, new_band)

            if training:
                reward = reward_for(new_band)

                old_q = qtable[state][action]
                target = reward + GAMMA * max(qtable[new_state])
                qtable[state][action] = old_q + ALPHA * (target - old_q)
                visits[state] += 1

                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            if len(trace) < TRACE_LIMIT:
                trace.append((steps, new_reflection, new_band, int(dark_mm),
                              tape_side, action, state, new_state))

            state = new_state
            band = new_band
            reflection = new_reflection
            steps += 1

            if training and steps % SAVE_EVERY_STEPS == 0:
                save_qtable(qtable)

            # Safety, not rescue. The lost penalty above has been charged for
            # every one of these steps; we simply stop before the robot leaves
            # the table, and wait for a human.
            if dark_mm >= LOST_HALT_MM:
                halt()
                ev3.speaker.beep(300, 400)
                show("LOST - halted", "", "Put it back on", "the edge, then",
                     "press CENTER")
                wait_press(Button.CENTER)
                ev3.speaker.beep(1000, 150)
                # A hand placement can put the tape on either side of the
                # sensor, so the belief has to be re-measured, not resumed.
                tape_side = probe_tape_side(tape_side)
                resync()

        except PauseException:
            halt()
            wait_for_release()
            ev3.speaker.beep(400, 150)

            show("PAUSED", "", "CENTER = resume")
            print("PAUSED. CENTER to resume.")
            wait_press(Button.CENTER)

            ev3.speaker.beep(1000, 150)
            resync()
            continue

finally:
    halt()

    if training:
        save_qtable(qtable)

    show("STOPPED", "", "steps: %d" % steps)

    print("TRACE step refl band darkmm side action state->next")
    for row in trace:
        print("T", row[0], row[1], BAND_NAMES[row[2]], row[3],
              SIDE_NAMES[row[4]], ACTION_NAMES[row[5]], row[6], row[7])
    print("TRACE rows:", len(trace), "of", steps, "steps")

    print("calibration black/edge/white", calibration,
          "DARK <", DARK_CUT, "LIGHT >", LIGHT_CUT)

    print("Q-table after", steps, "steps:")
    for index in range(STATE_COUNT):
        row = qtable[index]
        best = 0
        for a in range(1, ACTION_COUNT):
            if row[a] > row[best]:
                best = a
        rest = sorted(row)[-2]
        print(index,
              SIDE_NAMES[index // BAND_COUNT],
              BAND_NAMES[index % BAND_COUNT],
              ["%.1f" % v for v in row],
              "best", ACTION_NAMES[best],
              "margin %.1f" % (row[best] - rest),
              "n", visits[index])

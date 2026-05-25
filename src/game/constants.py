import math

# -- Game Constants --
MAX_STEPS = 500
MAX_PLANETS = 60  # (40 planets (5-10 symmetric groups of 4) + 5 groups of 4 comet spawns throughout the match)
TOTAL_COMETS = 20
ACT_TIMEOUT_SEC = 1.0
SUN_RADIUS = 10.0

# -- Board Constants --
BOARD_SIZE = 100.0
BOARD_CENTER = (50.0, 50.0)

# -- Planet Constants --
NEUTRAL_OWNER_ID = 0
PLAYER_OWNER_IDS = [1, 2, 3]
MAX_PRODUCTION = 5
MAX_RADIUS = 1 + math.log(MAX_PRODUCTION)
ROTATION_RADIUS_LIMIT = 50.0
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
COMET_RADIUS = 1.0  # fixed according to orbit wars docs
COMET_PRODUCTION = 1  # when owned
COMET_SPEED = 4.0  # according to orbit wars docs

# -- Fleet Constants --
MAX_FLEET_SPEED = 6.0
# This is how fleet launch positions are calculated:
# start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
# start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
PLANET_LAUNCH_RADIUS_OFFSET = 0.1

# -- Action Constants --
NO_OP_CANDIDATE_INDEX = 0
MAX_OWNER_FEATURE_PLAYERS = 4

# -- Feature base dims --
BASE_SELF_FEATURE_DIM = 30
BASE_CANDIDATE_FEATURE_DIM = 24
BASE_GLOBAL_FEATURE_DIM = 45
BASE_PLANET_FEATURE_DIM = 13
BASE_EDGE_FEATURE_DIM = 12
BASE_GLOBAL_FEATURE_V2_DIM = 46
ANGULAR_VELOCITY_NORM = 0.05

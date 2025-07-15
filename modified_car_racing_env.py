__credits__ = ["Andrea PIERRÉ"]

# Modifications:
# - Added feature-vector output (11D) for CarRacing environment
# - Features: (off_track_flag, speed, acceleration, angular velocity, angular acceleration,
#    sine of car angle relative to road trajectory, cosine of car angle relative to road trajectory,
#    distance to left edge of track, distance to center of track, distance to right edge of track,
#    unused placeholder (always zero))
# - Modified by Nathan, 2025

#The first commit of this file is the original work from the gymnasium library
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_racing.py
#To see my modifications look at the diffs from the first to the second commit of this file

import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

import shapely.geometry
from collections import deque

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: breaking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer left
    - 2: steer right
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` converts the environment to use discrete action space.
     The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    ```

    ## Version History
    - v2: Change truncation to termination when finishing the lap (1.0.0)
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode
        
        #used to track acceleration, angular velociy and angular acceleration
        self.speed_over_frames = deque(maxlen=2)
        self.speed_over_frames.append(0.0)
        self.angle_over_frames = deque(maxlen=3)
        #achieved will be set to a vector of length equal to number of track tiles
        # and will be used to track covered track tiles
        self.achieved = None
        #track reward
        self.my_reward = 0.0
        #we use this to store just the vertices of track tiles, without kerbstones
        self.track_tile_vertices = []

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.track_tile_vertices.append([road1_l, road1_r, road2_r, road2_l])
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        self.achieved = None
        self.my_reward = 0.0
        self.track_tile_vertices = []
        self.speed_over_frames = deque(maxlen=2)
        self.speed_over_frames.append(0.0)
        self.angle_over_frames = deque(maxlen=3)

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            action = np.asarray(action)
            action = action.astype(np.float64)
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                action = int(action)
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.my_reward = 0
        self.state, data = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Termination due to finishing lap
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100

        if self.render_mode == "human":
            self.render()

        info["data"] = data
        info["completed"] = self.achieved.sum()/len(self.road)

        self.my_reward -= 0.1
        return self.state, self.my_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    #distance between a point and a line
    @staticmethod
    def point_2_line_distance(p, a, b):
            if a == b:
                return math.hypot(p[0] - a[0], p[1] - a[1])
            numerator = abs((b[1]-a[1])*p[0] - (b[0]-a[0])*p[1] + b[0]*a[1] - b[1]*a[0])
            denominator = math.hypot(b[0] - a[0], b[1] - a[1])
            return numerator / denominator

    @staticmethod
    def normalize_angle(angle):
            #normalize angle between -pi and pi
            return math.atan2(math.sin(angle), math.cos(angle))

    def get_kinematics(self):
        acc = self.speed_over_frames[1] - self.speed_over_frames[0]
        angular_velocity_past = self.normalize_angle(self.angle_over_frames[1] - self.angle_over_frames[0])
        angular_velocity_present = self.normalize_angle(self.angle_over_frames[2] - self.angle_over_frames[1])
        angular_acceleration = (angular_velocity_present-angular_velocity_past)/2
        return acc, angular_velocity_present, angular_acceleration

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        data = self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        self._draw_point(self.surf, (self.car.hull.position[0], self.car.hull.position[1]), (255,255,255), zoom, trans, angle)

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
            return data
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H)), data
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H)), data
        else:
            return self.isopen, data

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        #used to track covererd track tiles
        if self.achieved is None:
            self.achieved = np.zeros((len(self.road_poly)))

        #store index of track tile the car is currently on
        #When not on a tile, it equals None
        current_polygon_index = None
        #information on track tile the car is closest to
        closest_tile = {"index": None, "dist": None, "center": None}

        car_pos = (self.car.hull.position[0], self.car.hull.position[1])
        car_point = shapely.geometry.Point(car_pos)
        track_tile_index = 0 #road poly contains both track tiles and kerbstones, so we use this to know what track tile we're on
        #Loop through all track tiles and find closest as well as current track tile, if on track
        for i, (poly, color) in enumerate(self.road_poly):
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
            self._draw_point(self.surf, poly[0], (255,255,255), zoom, translation, angle)
            if color!= [255,255,255] and color != [255,0,0]:
                is_kerbstone = False
            else:
                is_kerbstone = True
                continue

            #if the polgon is not a kerbstone, we check the distance
            polygon = shapely.geometry.Polygon(poly)
            dist = polygon.distance(car_point)
            if (closest_tile["index"] == None or dist < closest_tile["dist"]):
                closest_tile["index"] = track_tile_index
                closest_tile["dist"] = dist
                closest_tile["center"] = (polygon.centroid.x, polygon.centroid.y)
            #we check if we are inside the track tile
            if polygon.contains(car_point):
                current_polygon_index = track_tile_index
                if(self.achieved[i] == 0):
                    self.achieved[i] = 1
                    self.my_reward = 1000/len(self.track_tile_vertices)

            track_tile_index += 1

        #if off track, we want to use the closest tile to compute relative car position and angle
        off_track = False
        if(current_polygon_index==None):
            current_polygon_index = closest_tile["index"]
            off_track = True

        #this will hold the center of the current(or closest) track tile
        # and the track tile that is 5 tiles ahead
        key_points = []

        #color current tile (or closest tile, if outside track) black
        poly = self.track_tile_vertices[current_polygon_index]
        self._draw_colored_polygon(self.surf, poly, (0,0,0), zoom, translation, angle)
        self._draw_point(self.surf, closest_tile["center"], (255,255,255), zoom, translation, angle)
        key_points.append(closest_tile["center"])

        #color 5th tile from current tile blue
        poly_next = self.track_tile_vertices[(current_polygon_index+5)%len(self.track_tile_vertices)]
        self._draw_colored_polygon(self.surf, poly_next, (0,0,255), zoom, translation, angle)
        polygon_center = shapely.geometry.Polygon(poly_next).centroid
        self._draw_point(self.surf, (polygon_center.x, polygon_center.y), (255,255,255), zoom, translation, angle)
        key_points.append((polygon_center.x, polygon_center.y))

        colours = [(255,255,255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

        c = poly
        #midpoint of the right edge of the track
        midpoint_right = ((c[1][0]+c[2][0])/2, (c[1][1]+c[2][1])/2)
        self._draw_point(self.surf, midpoint_right, colours[0], zoom, translation, angle)
        
        #midpoint of the left edge of the track
        midpoint_left = ((c[0][0]+c[3][0])/2, (c[0][1]+c[3][1])/2)
        self._draw_point(self.surf, midpoint_left, colours[1], zoom, translation, angle)

        #midpoint of the top edge of the track
        midpoint_top = ((c[0][0]+c[1][0])/2, (c[0][1]+c[1][1])/2)
        self._draw_point(self.surf, midpoint_top, colours[2], zoom, translation, angle)

        #midpoint of the bottom edge of the track
        midpoint_bottom = ((c[2][0]+c[3][0])/2, (c[2][1]+c[3][1])/2)
        self._draw_point(self.surf, midpoint_bottom, colours[3], zoom, translation, angle)

        #distance to the right edge of the track
        dist_2_right_track = self.point_2_line_distance(car_pos, c[1], c[2])
        #distance to the left edge of the track
        dist_2_left_track = self.point_2_line_distance(car_pos, c[0], c[3])
        #distance to the center line of the track
        dist_2_center_track = self.point_2_line_distance(car_pos, midpoint_top, midpoint_bottom)

        #ensure that the distance to the center line is signed
        if(dist_2_left_track < dist_2_right_track):
            dist_2_center_track *= -1

        #normalize distance based on track width
        # norms should be equal as long the sides of the track are parallel but we'll just leave the conditional in
        if off_track:
            norm = math.hypot(midpoint_right[0] - midpoint_left[0], midpoint_right[1] - midpoint_left[1])
        else:
            norm = dist_2_left_track + dist_2_right_track
        dist_2_left_track /= norm
        dist_2_center_track /= norm
        dist_2_right_track /= norm

        car_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        self.speed_over_frames.append(car_speed)
        car_angle = self.normalize_angle(self.car.hull.angle)
        self.angle_over_frames.append(car_angle)

        #if we only have one angle, fill up the deque
        if(len(self.angle_over_frames) == 1):
            self.angle_over_frames.append(car_angle)
            self.angle_over_frames.append(car_angle)

        #get acceleration, angular velocity and angular acceleration using the deques
        acc, ang_vel, ang_acc = self.get_kinematics()

        cos_theta = np.cos(-car_angle)
        sin_theta = np.sin(-car_angle)

        self._draw_line(self.surf, key_points[0], key_points[1], (0,255,0), zoom, translation, angle)

        #key points (current tile center point, 5th tile ahead, center point) rotated to car's frame
        key_points_cf = [[p[0]*cos_theta - p[1]*sin_theta,
                                p[0]*sin_theta + p[1]*cos_theta] 
                                for p in key_points]
        #get angle of line between the two points
        line_angle = math.atan2(key_points_cf[1][1]-key_points_cf[0][1], key_points_cf[1][0]-key_points_cf[0][0])
        #it turns out that we need to subtract 90 degrees from the line angle so the car is facing forward
        #this is the angle of the road trajectory relative to the car
        car_relative_angle = self.normalize_angle(line_angle - math.radians(90))
        car_relative_sin = math.sin(car_relative_angle)
        car_relative_cos = math.cos(car_relative_angle)
        
        #the trailing 0 is unneccessary, I was intending on using it but it turns out that we don't need it
        return (int(off_track), car_speed, acc, ang_vel, ang_acc, car_relative_sin, car_relative_cos, 
                dist_2_left_track, dist_2_center_track, dist_2_right_track, 0)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _draw_point(
        self, surface, point, color, zoom, translation, angle, clip=True
    ):
        point = pygame.math.Vector2(point).rotate_rad(angle)
        point[0] = point[0] * zoom + translation[0]
        point[1] = point[1] * zoom + translation[1]
        if not clip or not (
            -MAX_SHAPE_DIM <= point[0] <= WINDOW_W + MAX_SHAPE_DIM and
            -MAX_SHAPE_DIM <= point[1] <= WINDOW_H + MAX_SHAPE_DIM
        ):
            return
        
        gfxdraw.aacircle(surface, int(point[0]), int(point[1]), 2, color)
        gfxdraw.filled_circle(surface, int(point[0]), int(point[1]), 2, color)

    def _draw_line(
        self, surface, point1, point2, color, zoom, translation, angle, clip=True
    ):
        p1 = pygame.math.Vector2(point1).rotate_rad(angle)
        p2 = pygame.math.Vector2(point2).rotate_rad(angle)

        p1[0] = p1[0] * zoom + translation[0]
        p1[1] = p1[1] * zoom + translation[1]

        p2[0] = p2[0] * zoom + translation[0]
        p2[1] = p2[1] * zoom + translation[1]

        if clip:
            if (
                p1[0] < -MAX_SHAPE_DIM and p2[0] < -MAX_SHAPE_DIM
                or p1[0] > WINDOW_W + MAX_SHAPE_DIM and p2[0] > WINDOW_W + MAX_SHAPE_DIM
                or p1[1] < -MAX_SHAPE_DIM and p2[1] < -MAX_SHAPE_DIM
                or p1[1] > WINDOW_H + MAX_SHAPE_DIM and p2[1] > WINDOW_H + MAX_SHAPE_DIM
            ):
                return  # Both points are offscreen, skip drawing

        pygame.gfxdraw.line(surface, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()

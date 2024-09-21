"""
Origin: https://github.com/Sebastian-Griesbach/MonsterCliffWalking
"""

from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
from collections import Counter
from collections import deque
from math import pow, floor, sqrt

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
HOLD = 4


class MonsterCliffWalkingEnv(Env):
    """
    Monster cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff and meeting a Monster.

    ## Description
    The game starts with the player at location [3, 0] of the 4x12 grid world with the
    goal located at [3, 11]. If the player reaches the goal the episode ends. The Monster may starts on any square on
    the upper 3x12 cells.

    A cliff runs along [3, 1..10]. If the player moves to a cliff location it
    returns to the start location. If the player encounters the Monster the episodes terminates with a huge penalty.

    The player makes moves until they reach the goal. At each step the player takes the Monster randomly moves in any direction (not diagonals)
    or stands still. The Monster only moves within the upper 3x12 cells.

    Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].

    Code adapted from:
    [https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py)

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left

    ## Observation Space
    There are 4*12 * 3*12  possible states. All possible player states multiplied by all possible monster states.

    The observation is an integer value representing the current combination of player and monster position as
    current_row * nrows + current_col for both entities and than using the Szudzik pairing function and a simple dictionary mapping to make
    the state indices dense.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in location [3, 0] and the monster anywhere in its 3x12 grid.

    ## Reward
    Each time step incurs -1 reward, unless the player stepped into the cliff,
    which incurs -100 reward.
    Encountering the monster incurs -1000 reward.

    ## Episode End
    The episode terminates when the player enters location [3, 11] or encounters the monster.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - "prob" - transition probability for the state.

    Monster cliff walking is stochastic, as the movement of the monster is random.

    ## Arguments

    ```python
    import gymnasium as gym

    gym.register(
        id='MonsterCliffWalking-v0',
        entry_point='path/to/module/monstercliffwalking:MonsterCliffWalkingEnv'
    )

    gym.make('MonsterCliffWalking-v0')
    ```

    ## References
    <a id="cliffwalk_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    ## Version History
    - v0: Initial version release

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
        "window_scaling": 2
    }

    def __init__(self, render_mode: Optional[str] = None):

        # Build bijective mapping from environment states to gapless int range
        self.shape = (4, 12)
        self.nPlayerStates = np.prod(self.shape)
        self.monster_state_shape = (self.shape[0] - 1, self.shape[1])
        self.nMonsterStates = np.prod(self.monster_state_shape)
        self.nS = self.nPlayerStates * self.nMonsterStates
        self.nA = 4
        szudzik_states = []
        for player_state in range(self.nPlayerStates):
            for monster_state in range(self.nMonsterStates):
                szudzik_states.append(pair(player_state, monster_state))

        self.state_to_szudzik = {state_id: szudzik_state for state_id, szudzik_state in enumerate(szudzik_states)}
        self.szudzik_to_state = {szudzik_state: state_id for state_id, szudzik_state in self.state_to_szudzik.items()}

        # initial settings
        self.player_start_state = int(np.ravel_multi_index((3, 0), self.shape))
        self.monster_start_state = int(np.ravel_multi_index((1, 6), self.shape))

        self.monster_facing = LEFT

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for state in range(self.nS):
            szudzik_state = self.state_to_szudzik[state]
            player_state, monster_state = unpair(szudzik_state, n=2)
            player_position = np.unravel_index(player_state, self.shape)
            monster_position = np.unravel_index(monster_state, self.monster_state_shape)
            self.P[state] = {a: [] for a in range(self.nA)}
            self.P[state][UP] = self._calculate_transition_prob(player_position, monster_position, [-1, 0])
            self.P[state][RIGHT] = self._calculate_transition_prob(player_position, monster_position, [0, 1])
            self.P[state][DOWN] = self._calculate_transition_prob(player_position, monster_position, [1, 0])
            self.P[state][LEFT] = self._calculate_transition_prob(player_position, monster_position, [0, -1])

        # Calculate initial state distribution
        self.initial_state_distrib = np.zeros(self.nS)

        start_state_indices = np.array(
            [self.szudzik_to_state[pair(self.player_start_state, monster_state)] for monster_state in
             range(self.nMonsterStates)])
        self.initial_state_distrib[start_state_indices] = 1 / start_state_indices.shape[0]

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        # pygame utils

        self.cell_size = (60 * self.metadata["window_scaling"], 60 * self.metadata["window_scaling"])
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.monster_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    # update the position of the monster
    def update_monster(self, action, current_monster_position):

        # get the current position of the monster
        i, j = current_monster_position
        # update the position according to the action
        if action == UP:
            i = max(i - 1, 0)
        elif action == DOWN:
            i = min(i + 1, self.monster_state_shape[0] - 1)
        elif action == RIGHT:
            j = min(j + 1, self.monster_state_shape[1] - 1)
        elif action == LEFT:
            j = max(j - 1, 0)

        return np.array([i, j])

    def calculate_monster_transition_probabilities(self, current_monster_position):
        # define the possible actions for the monster
        actions = [UP, DOWN, RIGHT, LEFT, HOLD]
        resulting_positions = []
        for action in actions:
            resulting_positions.append(tuple(self.update_monster(action, current_monster_position)))

        counts = dict(Counter(resulting_positions))
        probabilities = {position: count / len(actions) for position, count in counts.items()}

        return probabilities

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current_player_position, current_monster_position, delta):
        """Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current_player_position: Current player position on the grid as (row, col)
            current_monster_position: Current monster position on the grid as (row, col)
            delta: Change in position for player transition

        Returns:where both the row and col start at 0).
            Tuple of ``(probability, new_environment_state, reward, terminated)``
        """
        new_player_position = np.array(current_player_position) + np.array(delta)
        new_player_position = self._limit_coordinates(new_player_position).astype(int)
        new_player_state = int(np.ravel_multi_index(tuple(new_player_position), self.shape))

        monster_transition_probabilities = self.calculate_monster_transition_probabilities(current_monster_position)

        return_tuples = []
        for potential_monster_position, probability in monster_transition_probabilities.items():
            new_monster_state = int(np.ravel_multi_index(tuple(potential_monster_position), self.monster_state_shape))
            # player fall
            if self._cliff[tuple(new_player_position)]:
                return_tuples.append(
                    (probability, self.szudzik_to_state[pair(self.player_start_state, new_monster_state)], -100, False))

            # player meets the monster
            elif tuple(new_player_position) == tuple(potential_monster_position):
                return_tuples.append((probability, self.szudzik_to_state[pair(new_player_state, new_monster_state)],
                                      -1000, True))  # Yes he dies

            # normal move
            else:
                terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
                is_terminated = tuple(new_player_position) == terminal_state
                return_tuples.append(
                    (probability, self.szudzik_to_state[pair(new_player_state, new_monster_state)], -1, is_terminated))

        return return_tuples

    def step(self, a):
        old_monster_position = np.unravel_index(unpair(self.state_to_szudzik[self.s], n=2)[1], self.monster_state_shape)

        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        new_monster_position = np.unravel_index(unpair(self.state_to_szudzik[self.s], n=2)[1], self.monster_state_shape)
        if tuple(old_monster_position) != tuple(new_monster_position):
            if old_monster_position[0] < new_monster_position[0]:
                self.monster_facing = DOWN
            elif old_monster_position[0] > new_monster_position[0]:
                self.monster_facing = UP
            elif old_monster_position[1] > new_monster_position[1]:
                self.monster_facing = LEFT
            else:
                self.monster_facing = RIGHT

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # initial state
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": self.initial_state_distrib[self.s]}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("MonsterCliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "img/elf_up.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.monster_images is None:
            monsters = [
                path.join(path.dirname(__file__), "img/monster_up.png"),
                path.join(path.dirname(__file__), "img/monster_right.png"),
                path.join(path.dirname(__file__), "img/monster_down.png"),
                path.join(path.dirname(__file__), "img/monster_left.png"),
            ]
            self.monster_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in monsters
            ]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "img/mountain_bg1.png"),
                path.join(path.dirname(__file__), "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "img/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nPlayerStates):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if row < self.shape[0] - 1 and self._cliff[row + 1, col]:
                self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)
            if s == self.player_start_state:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nPlayerStates - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == unpair(self.state_to_szudzik[self.s], n=2)[0]:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        # Render Monster
        m = unpair(self.state_to_szudzik[self.s], n=2)[1]
        m_row, m_col = np.unravel_index(m, self.monster_state_shape)
        m_pos = (m_col * self.cell_size[0], m_row * self.cell_size[1])
        monster_pos = (m_pos[0], m_pos[1] - 0.1 * self.cell_size[1])
        facing = self.monster_facing if self.monster_facing is not None else 3
        self.window_surface.blit(self.monster_images[facing], monster_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        outfile = StringIO()

        for s in range(self.nPlayerStates):
            position = np.unravel_index(s, self.shape)
            if unpair(self.state_to_szudzik[self.s], n=2)[0] == s:
                output = " x "
            # Print Monster
            elif unpair(self.state_to_szudzik[self.s], n=2)[1] == s:
                output = " m "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()


# Following code taken from: https://github.com/Convert-Group/pairing-functions/blob/master/pairing_functions/szudzik.py
def pair(*numbers: int) -> int:
    """
    Maps a pair of non-negative integers to a uniquely associated single non-negative integer.
    Pairing also generalizes for `n` non-negative integers, by recursively mapping the first pair.
    For example, to map the following tuple:
    (n_1, n_2, n_3)
    the n_1, n_2 pair is mapped accordingly to a number n_p,
    and then the n_p, n3 pair is mapped to produce the final association.
    """
    if len(numbers) < 2:
        raise ValueError('Szudzik pairing function needs at least 2 numbers as input')

    elif any((n < 0) or (not isinstance(n, int)) for n in numbers):
        raise ValueError('Szudzik pairing function maps only non-negative integers')

    numbers = deque(numbers)

    # fetch the first two numbers
    n1 = numbers.popleft()
    n2 = numbers.popleft()

    if n1 != max(n1, n2):
        mapping = pow(n2, 2) + n1
    else:
        mapping = pow(n1, 2) + n1 + n2

    mapping = int(mapping)

    if not numbers:
        # recursion concludes
        return mapping
    else:
        numbers.appendleft(mapping)
        return pair(*numbers)


def unpair(number: int, n: int = 2) -> tuple:
    """
    The inverse function outputs the pair associated with a non-negative integer.
    Unpairing also generalizes by recursively unpairing a non-negative integer to `n` non-negative integers.
    For example, to associate a `number` with three non-negative
    integers n_1, n_2, n_3, such that:

    pairing(n_1, n_2, n_3) = `number`

    the `number` will first be unpaired to n_p, n_3, then the n_p will be unpaired to n_1, n_2,
    producing the desired n_1, n_2 and n_3.
    """
    if (number < 0) or (not isinstance(number, int)):
        raise ValueError('Szudzik unpairing function requires a non-negative integer')

    if number - pow(floor(sqrt(number)), 2) < floor(sqrt(number)):

        n1 = number - pow(floor(sqrt(number)), 2)
        n2 = floor(sqrt(number))

    else:
        n1 = floor(sqrt(number))
        n2 = number - pow(floor(sqrt(number)), 2) - floor(sqrt(number))

    n1, n2 = int(n1), int(n2)

    if n > 2:
        return unpair(n1, n - 1) + (n2,)
    else:
        # recursion concludes
        return n1, n2

# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# Skeleton from https://foozlecc.itch.io/lucifer-skeleton-grunt-enemy
# All other assets by ____

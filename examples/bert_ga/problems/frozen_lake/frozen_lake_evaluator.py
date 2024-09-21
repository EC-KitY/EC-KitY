"""
Origin: Frozen Lake Benchmark
https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#sphx-glr-tutorials-training-agents-frozenlake-tuto-py
"""

from typing import List

import gymnasium as gym

from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)

FROZEN_LAKE_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
FROZEN_LAKE_MAP_SIZE = len(FROZEN_LAKE_MAP)
HOLES = [
    i * FROZEN_LAKE_MAP_SIZE + j
    for i in range(FROZEN_LAKE_MAP_SIZE)
    for j in range(FROZEN_LAKE_MAP_SIZE)
    if FROZEN_LAKE_MAP[i][j] == "H"
]
FROZEN_LAKE_STATES = FROZEN_LAKE_MAP_SIZE ** 2 - len(HOLES) - 1


class FrozenLakeEvaluator(SimpleIndividualEvaluator):
    def __init__(
            self,
            arity=1,
            events=None,
            event_names=None,
            total_episodes=2000,
            is_slippery=True,
    ):
        super().__init__(arity, events, event_names)
        self.total_episodes = total_episodes

        # Generate a random 8x8 map with 80% of the cells being frozen
        # This map will remain the same through the whole evolutionary run
        map_size = FROZEN_LAKE_MAP_SIZE
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=f"{map_size}x{map_size}",
            is_slippery=is_slippery,
        )

    def evaluate_individual(self, individual):
        vector = individual.get_vector().copy()

        # Insert zeros for holes
        for hole in HOLES:
            vector.insert(hole, -1)

        score = 0
        for episode in range(self.total_episodes):
            state = self.env.reset()[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = self.choose_action(state=state, vector=vector)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, _ = self.env.step(
                    action
                )

                done = terminated or truncated

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            score += total_rewards

        return score / self.total_episodes

    def choose_action(self, state: int, vector: List[int]) -> int:
        """
        Returns the best action.
        """
        return int(vector[state])

    def terminate(self):
        self.env.close()

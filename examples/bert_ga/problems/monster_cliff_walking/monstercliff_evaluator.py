import gymnasium as gym
import numpy as np
from eckity.evaluators import SimpleIndividualEvaluator
from .monstercliffwalking import unpair

# player position: 4x12, monster position: 3x12
MONSTER_CLIFF_STATES = 1332  # 4*12*3*12 - 1*11*3*12


class MonsterCliffWalkingEvaluator(SimpleIndividualEvaluator):
    def __init__(
            self, arity=1, events=None, event_names=None, total_episodes=1000
    ):
        super().__init__(arity, events, event_names)
        self.total_episodes = total_episodes

        gym.register(
            id="MonsterCliffWalking-v0",
            entry_point="problems.monster_cliff_walking.monstercliffwalking:MonsterCliffWalkingEnv",
        )

        self.env = gym.make("MonsterCliffWalking-v0")

    def evaluate_individual(self, individual):
        vector = individual.get_vector().copy()

        score_sum = 0
        for episode in range(self.total_episodes):
            score = 0
            obs = self.env.reset()[0]
            done = False
            n_steps = 0

            while not done:
                action = self.choose_action(obs, vector)

                # Take the action and observe the outcome state and reward
                obs, reward, terminated, truncated, _ = self.env.step(action)

                n_steps += 1
                score += reward
                done = terminated or truncated or n_steps == 1000

            score_sum += score

        return score_sum / self.total_episodes

    def choose_action(self, obs: int, vector: list[int]) -> int:
        """
        Returns the best action.
        """
        number = self.env.unwrapped.state_to_szudzik[obs]
        state = unpair(number, n=2)
        idx = np.ravel_multi_index(state, MONSTER_CLIFF_SPACE_SHAPE)  # todo MONSTER_CLIFF_SPACE_SHAPE is not defined
        return int(vector[idx])

    def terminate(self):
        self.env.close()

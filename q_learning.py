import constants
import matplotlib.pyplot as plt
import numpy as np


class QLearning(object):

    def __init__(
        self,
        environment,
        learning_rate=None,
        discounting_factor=None,
        epsilon=None,
        show_success_epochs=False
    ):
        self.env = environment
        self.q_table = np.random.uniform(
            low=0, high=1, size=([30, 30, 50, 50] + [self.env.action_space.n]))
        self.learning_rate = learning_rate if learning_rate else constants.LEARNING_RATE
        self.discounting_factor = discounting_factor if discounting_factor else constants.DISCOUNTING_FACTOR
        self.epsilon = epsilon if epsilon else constants.EPSILON

        if not epsilon:
            self.epsilon_min = constants.EPSILON_MIN
            self.epsilon_decay = constants.EPSILON_DECAY

        self.steps_over_epochs = []

        self.show_success_epochs = show_success_epochs

    def learn(self, epochs=None):
        if not epochs:
            epochs = constants.NUMBER_OF_EPOCHS

        succeed = False
        epochs = 1
        times_of_success = 0
        epochs_after_success = 0

        # for epoch in range(1, epochs + 1):
        while not succeed:
            current_state = self.get_discrete_state(self.env.reset())
            for step in range(constants.MAX_STEPS_NUMBER):
                action = self.get_action_with_epsilon_greedy(current_state)
                new_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -2

                new_discrete_state = self.get_discrete_state(new_state)
                self.update_table(current_state, new_discrete_state, action, reward)

                if done:
                    self.steps_over_epochs.append(step)
                    break

                current_state = new_discrete_state

            # if epochs == 100000:
            #     self.epsilon = constants.EPSILON
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

            if epochs >= constants.BATCH_SIZE:
                steps_over_epochs_size = len(self.steps_over_epochs)
                list_of_steps = self.steps_over_epochs[
                                    (steps_over_epochs_size - constants.BATCH_SIZE):steps_over_epochs_size
                                ]

                if (sum(list_of_steps) / constants.BATCH_SIZE) >= 195:
                    times_of_success += 1
                    print(f'Did it successfully at epoch {epochs}')

            if times_of_success > 0:
                epochs_after_success += 1

            if epochs_after_success == 10000:
                succeed = True

            if epochs % 10000 == 0:
                print('...')

            epochs += 1

        print(f'Number of success: {times_of_success}')
        self.save_knowledge()
        self.plot_result()
        self.env.close()

    def play(self):
        try:
            done = False
            steps = 0

            with open('knowledge.txt', 'rb') as file:
                self.q_table = np.load(file)

            current_state = self.get_discrete_state(self.env.reset())

            while not done:
                self.env.render()
                action = np.argmax(self.q_table[current_state])
                new_state, _, done, _ = self.env.step(action)
                current_state = self.get_discrete_state(new_state)
                steps += 1

            print(f'End of game. You achieve {steps} steps')
            self.env.close()

            return steps
        except OSError:
            print('Can not find knowledge file!')
            quit()

    @staticmethod
    def get_discrete_state(state):
        discrete_state = state / np.array([0.25, 0.25, 0.01, 0.1]) + np.array([15, 10, 1, 10])
        return tuple(discrete_state.astype(int))

    def get_action_with_epsilon_greedy(self, current_state):
        random_number = np.random.rand()

        return (np.random.randint(0, self.env.action_space.n) if random_number < self.epsilon else
                np.argmax(self.q_table[current_state]))

    def update_table(self, current_state, new_state, action, reward):
        current_state_q_value = self.q_table[current_state + (action,)]
        max_new_state_q_value = np.max(self.q_table[new_state])
        q_value = current_state_q_value + self.learning_rate * (reward + self.discounting_factor * max_new_state_q_value - current_state_q_value)

        self.q_table[current_state + (action,)] = q_value

    def save_knowledge(self):
        with open('knowledge.txt', 'wb') as file:
            np.save(file, self.q_table)

    def plot_result(self):
        x_ = range(1, len(self.steps_over_epochs) + 1)
        y_ = self.steps_over_epochs
        plt.plot(x_, y_)
        plt.xlabel('epochs')
        plt.ylabel('steps')
        plt.title('Number of steps over epochs with Q-learning')
        plt.show()

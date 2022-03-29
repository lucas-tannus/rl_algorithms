import random
import cv2
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from os import path


RENDER                        = True
STARTING_EPISODE              = 951
ENDING_EPISODE                = 1000
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5


def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state


def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))


class CarRacingDQNAgent:
    def __init__(
        self,
        action_space    = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ],
        frame_stack_num = 3,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 0.3,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)


class DQN:

    def train(self):
        env = gym.make('CarRacing-v0')
        agent = CarRacingDQNAgent(epsilon=0.01)
        agent.load(path.join('save', 'trial_950.h5'))

        for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
            init_state = env.reset()
            init_state = process_state_image(init_state)

            total_reward = 0
            negative_reward_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1
            done = False

            while True:
                if RENDER:
                    env.render()

                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)

                reward = 0
                for _ in range(SKIP_FRAMES + 1):
                    next_state, r, done, info = env.step(action)
                    reward += r
                    if done:
                        break

                # If continually geprocess_state_imagetting negative reward 10 times after the tolerance steps, terminate this episode
                negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

                # Extra bonus for the model if it uses full gas
                if action[1] == 1 and action[2] == 0:
                    reward *= 1.5

                total_reward += reward

                next_state = process_state_image(next_state)
                state_frame_stack_queue.append(next_state)
                next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

                agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

                if done or negative_reward_counter >= 25 or total_reward < 0:
                    print(
                        'Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(
                            e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                    break
                if len(agent.memory) > TRAINING_BATCH_SIZE:
                    agent.replay(TRAINING_BATCH_SIZE)
                time_frame_counter += 1

            if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
                agent.update_target_model()

            if e % SAVE_TRAINING_FREQUENCY == 0:
                agent.save('./save/trial_{}.h5'.format(e))

        env.close()

    @staticmethod
    def play_with_model():
        env = gym.make('CarRacing-v0')
        agent = CarRacingDQNAgent(epsilon=0)

        for a in range(0, 100):
            agent.load(path.join('save', f'trial_1000.h5'))
            achieve = True

            for e in range(0, 100):
                init_state = env.reset()
                init_state = process_state_image(init_state)

                total_reward = 0
                punishment_counter = 0
                state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
                time_frame_counter = 1

                while True:
                    # env.render()
                    current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                    action = agent.act(current_state_frame_stack)
                    next_state, reward, done, info = env.step(action)

                    total_reward += reward

                    next_state = process_state_image(next_state)
                    state_frame_stack_queue.append(next_state)

                    if done:
                        achieve = achieve and total_reward >= 900
                        # print('Episode: {}/{} -> {}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(
                        #     e + 1, 1000, time_frame_counter, float(total_reward)))
                        # rewards.append(str(float(total_reward)))
                        break
                    time_frame_counter += 1
            with open('rewards.txt', 'a') as file:
                file.write('1' if achieve else '0')

                # with open('rewards.txt', 'a') as file:
            #     print(f'Writing on file for knowledge {a}')
            #     file.write(f'{",".join(rewards)}\n')

DQN().play_with_model()
# DQN().train()

for _ in range(1000):
    with open('rewards.txt', 'a') as file:
        file.write(f'{random.randrange(889, 1015)}, ')

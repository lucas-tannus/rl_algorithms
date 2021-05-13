import gym
import constants
import matplotlib.pyplot as plt

from q_learning import QLearning


def train(agent, epochs=None):
    if epochs is None:
        epochs = [constants.NUMBER_OF_EPOCHS]

    print(f'Starting agent training...')
    q_learning_agent = agent.learn(epochs)
    print(f'End o training')
    return q_learning_agent


def start():

    finish = False
    agent = QLearning(environment=gym.make('CartPole-v1'))
    games_results = []

    while not finish:
        cmd = str(input('''Q-learning
        1 - Train
        2 - Play
        3 - Plot results
        4 - Number of games that achieve more than 195 steps
        Other - Exit
        
        Enter with a command: '''))
        if cmd == '1':
            params = {}
            epochs_answer = str(input('How many epochs the agent will train: '))
            if epochs_answer != 'N':
                params['epochs'] = int(epochs_answer)

            train(agent, **params)
        elif cmd == '2':
            rounds = int(input(f'How many times do you want to play with agent: '))

            for _ in range(rounds):
                number_of_steps = agent.play()
                games_results.append(number_of_steps)
        elif cmd == '3':
            x_ = range(1, len(games_results) + 1)
            y_ = games_results
            plt.plot(x_, y_)
            plt.xlabel('games')
            plt.ylabel('number of steps')
            plt.title('Number of steps for each game')
            plt.show()
        elif cmd == '4':
            print(f'{len([steps for steps in games_results if steps > 195])} games achieve more than 195 steps')
        else:
            print('Ending...')
            quit()


if __name__ == '__main__':
    # start()
    env = gym.make('CarRacing-v0')
    print(env.reset())

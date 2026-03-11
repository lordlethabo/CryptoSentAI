import numpy as np
import random


class QLearningTrader:

    def __init__(self, actions=["BUY", "SELL", "HOLD"],
                 alpha=0.1,
                 gamma=0.9,
                 epsilon=0.1):

        self.q_table = {}

        self.actions = actions

        self.alpha = alpha

        self.gamma = gamma

        self.epsilon = epsilon


    def get_state(self, sentiment, trend, confidence):

        sentiment_bin = round(sentiment, 1)

        confidence_bin = round(confidence, 1)

        state = (sentiment_bin, trend, confidence_bin)

        return state


    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.choice(self.actions)

        if state not in self.q_table:

            self.q_table[state] = [0, 0, 0]

        action_index = np.argmax(self.q_table[state])

        return self.actions[action_index]


    def update(self, state, action, reward, next_state):

        if state not in self.q_table:

            self.q_table[state] = [0, 0, 0]

        if next_state not in self.q_table:

            self.q_table[next_state] = [0, 0, 0]

        action_index = self.actions.index(action)

        best_next = np.max(self.q_table[next_state])

        self.q_table[state][action_index] += self.alpha * (

            reward +

            self.gamma * best_next -

            self.q_table[state][action_index]

        )

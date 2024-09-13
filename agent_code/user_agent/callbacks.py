from agent_code.q_learning.features import Feature

def setup(self):
    self.feature = Feature()


def act(self, game_state: dict):
    print(self.feature(game_state))
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']

from agent_code.dqn.features import Feature

def setup(self):
    self.feature = Feature()


def act(self, game_state: dict):
    self.feature(game_state)
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']

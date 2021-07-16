import abc


class Policy(object):
    """
    General policy interface.
    """
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy):
    def set_num_steps_total(self, t):
        pass

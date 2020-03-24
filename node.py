import random


class Node:
    def __init__(self):
        self._id = str(random.random()) + str(random.random())
        self.is_fully_expanded = False
        self.is_terminal = False
        self.reward = 0
        self.parent = None
        self.children = []

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def is_subgoal(self):
        pass  # TODO

    def __repr__(self):
        return '[%s|-c:%d-|-r:%d-%s-|-%d-%d:%d-]' % (
            self._id,
            len(self.children),
            self.reward,
            str(self.action),
            self.env.agent_pos[0],
            self.env.agent_pos[1],
            self.env.agent_dir
        )

    def calc_hash(self):
        """This is just a default example implementation and needs to be adapted
        for each environment.
        Calculate and store a hash of the node that represents it.
        In this case, simply agent position and direction (since we are in a
        grid world) suffices.
        """
        # Calculate the hash using Cantor pairing and cache it.
        p = self.env.agent_pos
        d = self.env.agent_dir
        p = p[0] * 10 + p[1]
        self._hash = int(.5 * (p + d) * (p + d + 1) + d)

    def __hash__(self):
        # Return a cached hash instead of recalculating each time.
        # Improves performance by factor of 1.5.
        return self._hash

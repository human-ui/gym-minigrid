import numpy as np

from gym_minigrid import entities


ATTRS = {
        'cell': ['visible', 'visited', 'empty'],
        'object_type': entities.OBJECTS,
        'object_color': entities.COLORS,
        'object_state': entities.Door.STATES,
        'agent': ['is_here', 'is_carrying'],
        'agent_state': entities.Agent.STATES,
        'carrying_type': [t for t in entities.OBJECTS if entities.make(t).can_pickup()],
        'carrying_color': entities.COLORS
        }
SKIP = ['cell.visited', 'agent.is_here', 'agent_state']


class Encoder(object):

    def __init__(self, observation=False):

        self.keys = []
        count = 0
        for attr, values in ATTRS.items():
            keys = [f'{attr}.{v}' for v in values]
            self.keys.extend(keys)

            inds = {k: i + count for i,k in enumerate(values)}
            setattr(self, attr, inds)  # ugly but gives fast access
            count += len(values)

        if observation:
            o = [i for i,k in enumerate(self.keys) if self._keep(k)]
            self.obs_inds = np.array(o)
        else:
            self.obs_inds = np.arange(len(self.keys))

        self._slices()

    def __len__(self):
        return len(self.obs_inds)

    def _slices(self):
        count = 0
        slices = {}
        for attr, values in ATTRS.items():
            if attr in ['cell', 'agent']:
                for value in values:
                    attr_name = f'{attr}.{value}'
                    if self._keep(attr_name):
                        slices[attr_name] = slice(count, count + 1)
                        count += 1
            else:
                if self._keep(attr):
                    slices[attr] = slice(count, count + len(values))
                    count += len(values)

        self.slices = slices

    def _keep(self, key):
        return key not in SKIP and key.split('.')[0] not in SKIP


class Decoder(Encoder):

    def __init__(self, array, observation=False):
        super().__init__(observation=observation)

        assert len(array) == len(self.obs_inds)

        if observation:
            self.skip = SKIP
        else:
            self.skip = []

        count = 0
        for attr, values in ATTRS.items():
            if attr not in self.skip:
                keys = []
                choices = []
                for v in values:
                    name = f'{attr}.{v}'
                    if name not in self.skip:
                        keys.append(v)
                        choices.append(array[count])
                        count += 1

                if attr in ['cell', 'agent']:  # argmax does not apply
                    # single choice: False if <.5, otherwise True
                    choice = {k: v >= .5 for k,v in zip(keys, choices)}
                else:
                    # max over multiple choices
                    choice = keys[np.argmax(choices)]

                setattr(self, attr, choice)  # a convenience

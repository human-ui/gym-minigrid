import numpy as np

from gym_minigrid import entities

ATTRS = ['visible',
         'empty',
         {'object_type': ['wall', 'door', 'key', 'ball', 'box', 'goal', 'lava']},
         {'object_color': ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey']},
         {'door_state': ['open', 'closed', 'locked']},
         'agent_pos',
         'carrying',
         {'agent_state': ['right', 'down', 'left', 'up']},
         {'carrying_type': ['key', 'ball', 'box']},
         {'carrying_color': ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey']}
         ]

# ATTRS = {
#         'cell': ['visible', 'visited', 'empty'],
#         'object_type': ['wall', 'door', 'key', 'ball', 'box', 'goal', 'lava'],
#         'object_color': ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey'],
#         'object_state': ['open', 'closed', 'locked'],
#         'agent': ['is_here', 'is_carrying'],
#         'agent_state': ['right', 'down', 'left', 'up'],
#         'carrying_type': ['key', 'ball', 'box'],
#         'carrying_color': ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey']
#         }
SKIP = ['agent_pos', 'right', 'down', 'left', 'up']


# class CONST(object):

#     VISIBLE = 0
#     VISITED = 1
#     EMPTY = 2


# class Indices(object):

#     # __slots__ = list(ATTRS.keys())

#     def __init__(self):
#         count = 0
#         for attr, values in ATTRS.items():
#             inds = list(range(count, count + len(values)))
#             setattr(self, attr, inds)  # ugly but gives fast access
#             count += len(values)
#         self.count = count

#     def __len__(self):
#         return len(self.count)


class Channels(object):

    # __slots__ = list(ATTRS.keys())

    def __init__(self):
        self.attrs = {}
        self.inds = {}
        count = 0
        for attr in ATTRS:
            if isinstance(attr, dict):
                key = list(attr.keys())[0]
                values = list(attr.values())[0]
                inds = {v: i + count for i,v in enumerate(values)}
                # if indices_only:
                inds = np.arange(count, count + len(values))

                for ind, value in zip(inds, values):
                    if key.startswith('carrying'):
                        k = f'carrying_{value}'
                    else:
                        k = value
                    setattr(self, k, ind)
                    self.attrs[k] = ind

                self.attrs[key] = values
                count += len(values)

            else:
                key = attr
                inds = count
                self.inds[key] = count
                count += 1
            setattr(self, key, inds)  # ugly but gives fast access

        self.count = count
        self.obs_inds = [v for k,v in self.inds.items() if k not in SKIP]

    def __len__(self):
        return self.count


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

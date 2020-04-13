import torch

ATTRS = ('visible',
         'empty',
         {'object_type': ('wall', 'door', 'key', 'ball', 'box', 'goal', 'lava')},
         {'object_color': ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey')},
         {'door_state': ('open', 'closed', 'locked')},
         'agent_pos',
         {'agent_state': ('right', 'down', 'left', 'up')},
         'carrying',
         {'carrying_type': ('key', 'ball', 'box')},
         {'carrying_color': ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey')}
         )
SKIP = ('agent_pos', 'right', 'down', 'left', 'up')


class Channels(object):

    def __init__(self, device='cpu'):
        """
        Gives access to:
        - Indices of every attribute: Channels.wall -> 2, Channels.visible -> 0
        - Indices of every group of attributes: Channels.object_type -> [2, 3, 4, 5, 6, 7, 8]
        - String names of attributes as a dict: Channels.attrs['object_type'] -> ['wall', 'door', 'key', 'ball', 'box', 'goal', 'lava']
        """
        self.device = device

        self.attrs = {}
        self.inds = {}
        count = 0
        for attr in ATTRS:
            if isinstance(attr, dict):
                key = list(attr.keys())[0]
                values = list(attr.values())[0]
                inds = {v: i + count for i,v in enumerate(values)}
                # if indices_only:
                inds = torch.arange(count, count + len(values), dtype=torch.long, device=self.device)

                for ind, value in zip(inds, values):
                    if key.startswith('carrying'):
                        k = f'carrying_{value}'
                    else:
                        k = value
                    setattr(self, k, ind)
                    self.inds[k] = ind

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

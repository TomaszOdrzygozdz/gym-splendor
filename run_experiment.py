from collections import namedtuple

import gin



@gin.configurable
class Pupu:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

gin.parse_config_file('config1.gin')


x = Pupu()
print(x.a)

dudek = namedtuple('wiki', )
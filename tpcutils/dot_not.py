import sys
import collections
import yaml


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value


# class MyConstructor(SafeConstructor):
#     def construct_mapping(self, node, deep=False):
#         res = SafeConstructor.construct_mapping(self, node, deep)
#         assert isinstance(res, dict)
#         return AttributeDict(**res)
#
#
# class MyLoader(Reader, Scanner, Parser, Composer, MyConstructor, Resolver):
#     def __init__(self, stream, version=None):
#         Reader.__init__(self, stream)
#         Scanner.__init__(self)
#         Parser.__init__(self)
#         Composer.__init__(self)
#         MyConstructor.__init__(self)
#         Resolver.__init__(self)

import argparse
import os
import re

import yaml


def setup_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', nargs='+', required=True)
    opt, unknown_args = parser.parse_known_args()
    config = merge_configs(*[Config(path) for path in opt.config])
    config.parse_args(unknown_args)
    config.parse_refs()

    config.output_dir = os.path.join(config.output_dir, config.exp_name)

    return config


class Config(dict):
    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super(Config, self).__init__()

        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('.yml'):
                    with open(arg, 'r') as f:
                        raw_dict = yaml.safe_load(f)
                else:
                    raise Exception('unknown file format %s' % arg)

                init_assign(config=self, d=raw_dict)
            elif isinstance(arg, dict):
                init_assign(config=self, d=arg)
            else:
                raise TypeError('arg should be an instance of <str> or <dict>')

        if kwargs:
            init_assign(config=self, d=kwargs)


    def __call__(
        self,
        *args,
        **kwargs,
    ):
        return Config(self, *args, **kwargs)


    def __repr__(
        self,
        indent=4,
        prefix='',
    ):

        r = []

        for key, value in sorted(self.items()):
            if isinstance(value, Config):
                r.append('{}{}:'.format(prefix, key))
                r.append(value.__repr__(indent, prefix + ' ' * indent))
            else:
                r.append('{}{}: {}'.format(prefix, key, value))

        return '\n'.join(r)


    def __setstate__(
        self,
        state,
    ):
        init_assign(config=self, d=state)


    def __getstate__(self):

        d = dict()

        for key, value in self.items():
            if type(value) is Config:
                value = value.__getstate__()

            d[key] = value

        return d


    def __getattr__(
        self,
        key,
    ):

        try:
            return self[key]
        except KeyError:
            return None


    def __setattr__(
        self,
        key,
        value,
    ):
        self[key] = value


    def __delattr__(
        self,
        key,
    ):
        del self[key]


    def __getitem__(
        self,
        key,
    ):

        sub_cfg, sub_key = consume_dots(config=self, key=key, create_default=False)
        return dict.__getitem__(sub_cfg, sub_key)


    def __setitem__(
        self,
        key,
        value,
    ):

        sub_cfg, sub_key = consume_dots(config=self, key=key, create_default=True)
        if sub_cfg.__contains__(sub_key) and value == '_DELETE_CONFIG_':
            dict.__delitem__(sub_cfg, sub_key)
        else:
            dict.__setitem__(sub_cfg, sub_key, value)


    def __delitem__(
        self,
        key,
    ):

        sub_cfg, sub_key = consume_dots(config=self, key=key, create_default=False)
        dict.__delitem__(sub_cfg, sub_key)


    def __contains__(
        self,
        key,
    ):

        try:
            sub_cfg, sub_key = consume_dots(config=self, key=key, create_default=False)
        except KeyError:
            return False

        return dict.__contains__(sub_cfg, sub_key)


    def all_items(
        self,
        only_leaf=True,
    ):

        for key, value in traverse_dfs(self,
            continue_type=Config,
            only_leaf=only_leaf):
            yield key, value


    def parse_args(
        self,
        cmd_args,
    ):

        index = 0
        while index < len(cmd_args):
            arg = cmd_args[index]
            err_msg = 'invalid command line argument pattern: %s' % arg
            assert arg.startswith('--'), err_msg
            assert len(arg) > 2, err_msg
            assert arg[2] != '-', err_msg

            arg = arg[2:]
            if '=' in arg:
                key, full_value_str = arg.split('=')
                index += 1
            else:
                assert len(
                    cmd_args) > index + 1, 'incomplete command line arguments'
                key = arg
                full_value_str = cmd_args[index + 1]
                index += 2

            if ':' in full_value_str:
                value_str, value_type_str = full_value_str.split(':')
                value_type = eval(value_type_str)
            else:
                value_str = full_value_str
                value_type = None

            if key not in self:
                raise KeyError(key)

            if value_type is None:
                value_type = type(self[key])

            if value_type is bool:
                self[key] = {
                    'true': True,
                    'True': True,
                    '1': True,
                    'false': False,
                    'False': False,
                    '0': False,
                }[value_str]
            else:
                self[key] = value_type(value_str)


    def parse_refs(
        self,
        subconf=None,
        stack_depth=1,
        max_stack_depth=10,
    ):

        if stack_depth > max_stack_depth:
            raise Exception(
                ('Recursively calling `parse_refs` too many times'
                    'with stack depth > {}. '
                    'A circular reference may exists in your config.\n'
                    'If deeper calling stack is really needed,'
                    'please call `parse_refs` with extra argument like: '
                    '`parse_refs(max_stack_depth=9999)`').format(max_stack_depth))

        if subconf is None:
            subconf = self

        for key in subconf.keys():
            value = subconf[key]
            if type(value) is str and '@' in value:
                if value.count('@') == 1 and value.startswith(
                    '@{') and value.endswith('}'):
                    ref_key = value[2:-1]
                    ref_value = self[ref_key]
                    subconf[key] = ref_value
                else:
                    ref_key_list = re.findall(pattern="'@{(.+?)}'", string=value)
                    ref_key_list = list(set(ref_key_list))
                    ref_value_list = [
                        self[ref_key] for ref_key in ref_key_list
                    ]
                    origin_ref_key_list = [
                        "'@{" + ref_key + "}'" for ref_key in ref_key_list
                    ]

                    for origin_ref_key, ref_value in zip(origin_ref_key_list, ref_value_list):
                        value = value.replace(origin_ref_key, str(ref_value))
                    subconf[key] = value

        for key in subconf.keys():
            value = subconf[key]
            if type(value) is Config:
                self.parse_refs(value, stack_depth + 1)


def merge_configs(
    *configs,
):

    final_config = Config()

    for i in range(len(configs)):
        config = configs[i]
        if not isinstance(config, Config):
            raise TypeError(
                'config.merge_configs expect `Config` type inputs, '
                'but got `{}`.\n'
                'Correct usage: merge_configs(config1, config2, ...)\n'
                'Incorrect usage: merge_configs([configs1, configs2, ...])'.
                format(type(config)))

        final_config = final_config(dict(config.all_items()))

    return final_config


def consume_dots(
    config,
    key,
    create_default,
):

    sub_keys = key.split('.', 1)
    sub_key = sub_keys[0]
    if sub_key in Config.__dict__:
        raise KeyError(
            '"{}" is a preserved API name, '
            'which should not be used as normal dictionary key'.format(
                sub_key))

    if not dict.__contains__(config, sub_key) and len(sub_keys) == 2:
        if create_default:
            dict.__setitem__(config, sub_key, Config())
        else:
            raise KeyError(key)

    if len(sub_keys) == 1:
        return config, sub_key
    else:
        sub_config = dict.__getitem__(config, sub_key)
        if type(sub_config) != Config:
            if create_default:
                sub_config = Config()
                dict.__setitem__(config, sub_key, sub_config)
            else:
                raise KeyError(key)
        return consume_dots(config=sub_config, key=sub_keys[1], create_default=create_default)


def traverse_dfs(
    root,
    continue_type,
    only_leaf,
    key_prefix='',
):

    for key, value in root.items():
        full_key = '.'.join([key_prefix, key]).strip('.')
        child_kvs = []

        if type(value) == continue_type:
            for kv in traverse_dfs(root=value, continue_type=continue_type, only_leaf=only_leaf,
                key_prefix=full_key):
                child_kvs.append(kv)

        if len(child_kvs
        ) == 0 or type(value) != continue_type or not only_leaf:
            yield full_key, value

        for kv in child_kvs:
            yield kv


def init_assign(
    config,
    d,
):

    for full_key, value in traverse_dfs(d,
        continue_type=dict,
        only_leaf=True):
        sub_cfg, sub_key = consume_dots(config=config, key=full_key, create_default=True)
        sub_cfg[sub_key] = value


def get_config(
    obj,
    name,
    default=None,
):

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)

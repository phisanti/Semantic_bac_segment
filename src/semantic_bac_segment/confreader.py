import yaml
import os
from typing import Any, Dict

class ConfReader:
    """
    Load yaml config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading yaml config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key'].
        """

        def __getattr__(self, key: str) -> Any:
            return self[key]

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    def __init__(self, conf_name: str) -> None:
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self) -> Dict[str, Any]:
        assert os.path.exists(self.conf_name), f"File {self.conf_name} not found"
        with open(self.conf_name, "r") as conf:
            opt = yaml.safe_load(conf)
        return opt

    def __get_opt(self) -> DictWithAttributeAccess:
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)
        return opt

    def pretty_print(self, d: Dict[str, Any], title: str = 'Settings', indent: int = 0) -> None:
        print('Training settings:')
        for key, value in d.items():
            print(' ' * indent + str(key) + ':', end=' ')
            if isinstance(value, dict):
                print()
                self.pretty_print(value, indent + 2)
            else:
                print(str(value))

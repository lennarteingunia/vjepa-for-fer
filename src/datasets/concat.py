import importlib
from typing import List
import lightning


class ConcatDataModule(lightning.LightningDataModule):

    def __init__(self, modules: List[dict]):
        super(ConcatDataModule, self).__init__()
        self.modules = [self.build_module(module_cfg)
                        for module_cfg in modules]

    def build_module(self, module_cfg: dict) -> lightning.LightningDataModule:
        package_name, _, class_name = module_cfg['class_path'].rpartition('.')
        module = importlib.import_module(package_name)
        cls = getattr(module, class_name)
        return cls(**module_cfg['init_args'])
    
    def  train_dataloader(self):
        return [module.train_dataloader() for module in self.modules]
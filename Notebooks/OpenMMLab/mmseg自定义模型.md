# 自定义模型

[官方教程](https://mmsegmentation.readthedocs.io/zh_CN/latest/tutorials/customize_models.html)

## 自定义主干网络

自定义 xception 主干网络：

1. 创建一个新文件 `mmseg/models/backbone/xception.py`

```python
import torch.nn as nn

from ..registry import BACKBONES


@BACKBONES.register_module
class Xception(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self):
        pass
```

2. 在 `mmseg/models/backbone/__init__.py` 中导入模块

```python
from .xception import Xception

__all__ = [
    'Xception',
]
```

3. 在配置文件中使用它

```python
model = dict(
    ...
    backbone=dict(
        type='Xception',
        arg1=xxx,
        arg2=xxx),
    ...
```

- [完整 Xception65 代码](../../Code/xception/xception.py)
- [DeepLabv3Plus-xception65-d8 配置文件](../../Code/xception/deeplabv3plus_xception-d8_513x513_60k_trimap.py)
- [DeepLabv3Plus-xception65-d16 配置文件](../../Code/xception/deeplabv3plus_xception-d16_513x513_60k_trimap.py)





<!-- 完成标志, 看不到, 请忽略! -->

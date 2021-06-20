import logging
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .utils import locate_candidate_layer

__all__ = ['_CAM']


class _CAM:
    """解释图提取器

    Args:
        model: 输入模型
        target_layer: 模型的名称
        input_shape: 输入的形状
    """

    def __init__(
            self,
            model: nn.Module,
            target_layer: Optional[str] = None,
            input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        # Obtain a mapping from module name to module instance for each layer in the model
        self.submodule_dict = dict(model.named_modules())

        # If the layer is not specified, try automatic resolution
        if target_layer is None:
            target_layer = locate_candidate_layer(model, input_shape)
            # 提醒用户设置目标层
            if isinstance(target_layer, str):
                logging.warning(f"no value was provided for `target_layer`, thus set to '{target_layer}'.")
            else:
                raise ValueError("unable to resolve `target_layer` automatically, please specify its value.")

        if target_layer not in self.submodule_dict.keys():
            raise ValueError(f"Unable to find submodule {target_layer} in the model")
        self.target_layer = target_layer
        self.model = model
        self.hook_a: Optional[Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        #前向传播时目标层的输出，用于获取特征层
        self.hook_handles.append(self.submodule_dict[target_layer].register_forward_hook(self._hook_a))
        #hook注册激活标志
        self._hooks_enabled = True
        #归一化之前是否使用relu激活函数
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """特征层获取"""
        if self._hooks_enabled:
            self.hook_a = output.data

    def clear_hooks(self) -> None:
        """清除RemovableHandle列表"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """解释图归一化，对于每个特征矩阵，其每个元素减去特征矩阵中的最小值之后再除于特征矩阵中的最大值"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])
        return cams

    def _get_weights(self, latent_pos: int, scores: Optional[Tensor] = None) -> Tensor:
        """在子类中实现"""
        raise NotImplementedError

    def _precheck(self, latent_pos: int, scores: Optional[Tensor] = None) -> None:
        """保证有效的计算实例"""
        #检查是否获取到特征矩阵
        if not isinstance(self.hook_a, Tensor):
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        # Check batch size
        if self.hook_a.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.hook_a.shape[0]}")
        # Check latent_pos value
        if not isinstance(latent_pos, int) or latent_pos < 0:
            raise ValueError("Incorrect `latent_pos` argument value")

        #检查是否使用分函数计算以及分函数是否被传入
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be passed to compute CAMs")

    def __call__(self, latent_pos: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:
        """调用"""
        # Integrity check
        self._precheck(latent_pos, scores)

        #计算解释图
        return self.compute_cams(latent_pos, scores, normalized)

    def compute_cams(self, latent_pos: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:
        """对特定位置的隐变量计算解释图
        Args:
            latent_pos (int): 待解释的隐变量
            scores (torch.Tensor[1, K], optional): 分函数的值
            normalized (bool, optional): 是否进行归一化的标志

        Returns:
            torch.Tensor[M, N]: 返回的解释图类型
        """

        # 计算权重因子 unsqueeze it
        weights = self._get_weights(latent_pos, scores)

        weights = weights[(...,) + (None,) * (self.hook_a.ndim - 2)]
        # type: ignore[operator, union-attr]

        #根据权重因子计算解释图
        batch_cams = torch.nansum(weights * self.hook_a.squeeze(0), dim=0)  # type: ignore[union-attr]
        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        #是否对解释图进行归一化
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams

    def extra_repr(self) -> str:
        return f"target_layer='{self.target_layer}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

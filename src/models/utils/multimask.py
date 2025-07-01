# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        if masks is None:
            return self.backbone(x)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs


class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs
    
class SLTPredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super(SLTPredictorMultiMaskWrapper, self).__init__()
        self._backbone = backbone

    def forward(self, img_ctxt, lang_ctxt, mask_tgt, masks_ctxt_indices, masks_tgt_indices):
        if type(img_ctxt) is not list:
            img_ctxt = [img_ctxt]
        if type(lang_ctxt) is not list:
            lang_ctxt = [lang_ctxt]
        if type(mask_tgt) is not list:
            masks_tgt = [masks_tgt]
        if type(masks_ctxt_indices) is not list:
            masks_ctxt_indices = [masks_ctxt_indices]
        if type(masks_tgt_indices) is not list:
            masks_tgt_indices = [masks_tgt_indices]

        outs = []
        for idx, (zi, li, hi, mc, mt) in enumerate(zip(img_ctxt, lang_ctxt, masks_tgt, masks_ctxt_indices, masks_tgt_indices)):
            outs += [self._backbone(zi, li, hi, mc, mt, mask_index=idx)]
        return outs

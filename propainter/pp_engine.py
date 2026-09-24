"""In-process ProPainter inference: models are loaded once and frames/masks are passed as numpy arrays
(same logic as inference_propainter.py)."""
import cv2
import numpy as np
import torch

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from inference_propainter import get_ref_index


class ProPainter:
    def __init__(self, weights="weights", device="cuda", fp16=True, raft_iter=20,
                 neighbor_length=10, ref_stride=10, subvideo_length=80):
        self.dev = torch.device(device)
        self.fp16 = fp16
        self.raft_iter = raft_iter
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride
        self.subvideo_length = subvideo_length
        self.raft = RAFT_bi(f"{weights}/raft-things.pth", self.dev)
        fc = RecurrentFlowCompleteNet(f"{weights}/recurrent_flow_completion.pth")
        for p in fc.parameters():
            p.requires_grad = False
        self.fc = fc.to(self.dev).eval()
        self.model = InpaintGenerator(model_path=f"{weights}/ProPainter.pth").to(self.dev).eval()
        if fp16:
            self.fc = self.fc.half()
            self.model = self.model.half()

    @torch.no_grad()
    def __call__(self, frames, masks, dilation=2):
        """frames: [HxWx3 BGR uint8] (H and W multiples of 8); masks: [HxW bool] -> [HxWx3 BGR uint8]"""
        dev = self.dev
        T = len(frames)
        h, w = frames[0].shape[:2]
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        m = np.stack([cv2.dilate(mk.astype(np.uint8), cross, iterations=dilation) if dilation else mk.astype(np.uint8)
                      for mk in masks])
        ori = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]  # RGB
        x = torch.from_numpy(np.stack(ori)).to(dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1).unsqueeze(0)
        masks_dilated = torch.from_numpy(m).to(dev).float()[None, :, None]
        flow_masks = masks_dilated

        # ---- optical flow (RAFT, fp32) ----
        short = 12 if w <= 640 else 8 if w <= 720 else 4 if w <= 1280 else 2
        if T > short:
            ff, fb = [], []
            for f in range(0, T, short):
                e = min(T, f + short)
                a, b = self.raft(x[:, f:e] if f == 0 else x[:, f - 1:e], iters=self.raft_iter)
                ff.append(a)
                fb.append(b)
            gt = (torch.cat(ff, 1), torch.cat(fb, 1))
        else:
            gt = self.raft(x, iters=self.raft_iter)
        torch.cuda.empty_cache()

        if self.fp16:
            x, flow_masks, masks_dilated = x.half(), flow_masks.half(), masks_dilated.half()
            gt = (gt[0].half(), gt[1].half())

        # ---- flow completion ----
        L = gt[0].size(1)
        sv = self.subvideo_length
        if L > sv:
            pf, pb = [], []
            pad = 5
            for f in range(0, L, sv):
                s, e = max(0, f - pad), min(L, f + sv + pad)
                ps, pe = f - s, e - min(L, f + sv)
                sub = (gt[0][:, s:e], gt[1][:, s:e])
                pred, _ = self.fc.forward_bidirect_flow(sub, flow_masks[:, s:e + 1])
                pred = self.fc.combine_flow(sub, pred, flow_masks[:, s:e + 1])
                pf.append(pred[0][:, ps:e - s - pe])
                pb.append(pred[1][:, ps:e - s - pe])
            pred_flows = (torch.cat(pf, 1), torch.cat(pb, 1))
        else:
            pred, _ = self.fc.forward_bidirect_flow(gt, flow_masks)
            pred_flows = self.fc.combine_flow(gt, pred, flow_masks)
        del gt
        torch.cuda.empty_cache()

        # ---- image propagation ----
        masked = x * (1 - masks_dilated)
        svp = min(100, sv)
        if T > svp:
            uf, um = [], []
            pad = 10
            for f in range(0, T, svp):
                s, e = max(0, f - pad), min(T, f + svp + pad)
                ps, pe = f - s, e - min(T, f + svp)
                b, t = masks_dilated[:, s:e].shape[:2]
                sub = (pred_flows[0][:, s:e - 1], pred_flows[1][:, s:e - 1])
                prop, upd = self.model.img_propagation(masked[:, s:e], sub, masks_dilated[:, s:e], "nearest")
                fr = x[:, s:e] * (1 - masks_dilated[:, s:e]) + prop.view(b, t, 3, h, w) * masks_dilated[:, s:e]
                uf.append(fr[:, ps:e - s - pe])
                um.append(upd.view(b, t, 1, h, w)[:, ps:e - s - pe])
            updated_frames, updated_masks = torch.cat(uf, 1), torch.cat(um, 1)
        else:
            b, t = masks_dilated.shape[:2]
            prop, upd = self.model.img_propagation(masked, pred_flows, masks_dilated, "nearest")
            updated_frames = x * (1 - masks_dilated) + prop.view(b, t, 3, h, w) * masks_dilated
            updated_masks = upd.view(b, t, 1, h, w)
        torch.cuda.empty_cache()

        # ---- feature propagation + transformer ----
        comp = [None] * T
        stride = self.neighbor_length // 2
        ref_num = sv // self.ref_stride if T > sv else -1
        bm_all = masks_dilated[0, :, 0].bool()
        for f in range(0, T, stride):
            nb = list(range(max(0, f - stride), min(T, f + stride + 1)))
            refs = get_ref_index(f, nb, T, self.ref_stride, ref_num)
            ids = nb + refs
            pflow = (pred_flows[0][:, nb[:-1]], pred_flows[1][:, nb[:-1]])
            pred = self.model(updated_frames[:, ids], pflow, masks_dilated[:, ids], updated_masks[:, ids], len(nb))
            pred = ((pred.view(-1, 3, h, w) + 1) / 2 * 255).float().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            bms = bm_all[nb].cpu().numpy()
            for j, idx in enumerate(nb):
                img = ori[idx].copy()
                img[bms[j]] = pred[j][bms[j]]
                if comp[idx] is None:
                    comp[idx] = img
                else:
                    comp[idx] = ((comp[idx].astype(np.float32) + img.astype(np.float32)) * 0.5).astype(np.uint8)
        torch.cuda.empty_cache()
        return [np.ascontiguousarray(c[:, :, ::-1]) for c in comp]

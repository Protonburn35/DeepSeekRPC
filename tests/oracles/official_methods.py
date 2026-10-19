# Extracted unchanged mathematical methods from the pinned official 1017 source.
# See PROVENANCE.json and LICENSE in this directory.
import torch
import torch.nn.functional as F
linear = F.linear

class Gate:

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
            scores = linear(x.float(), self.weight.float()) / self.gate_temp
            if self.score_func == "softmax":
                scores = scores.softmax(dim=-1)
            elif self.score_func == "sigmoid":
                scores = scores.sigmoid()
            else:
                scores = F.softplus(scores).sqrt()
            bias = self.bias
            if image_mask is not None and self.bias_vl is not None:
                bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
            # the bias picks experts but does not scale them: weights come from the raw scores
            indices = (scores + bias).topk(self.topk, dim=-1)[1]
            weights = scores.gather(1, indices)
            if self.norm_topk_prob and self.topk > 1:
                weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
            weights *= self.route_scale
            return weights, indices

class Expert:

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
            dtype = x.dtype
            gate = self.w1(x).float()
            up = self.w3(x).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            x = F.silu(gate) * up
            if weights is not None:
                x = weights * x
            return self.w2(x.to(dtype))

class Block:

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
            """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
            y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
            return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
            """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
            x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
            y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            return y.type_as(x)

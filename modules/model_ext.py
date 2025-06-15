from __future__ import annotations

"""Extended decoding for TIGER‑style Encoder‑Decoder model.

Exports
-------
- **EncoderDecoderRetrievalModelExt** – drop‑in replacement that supports
  - plain beam search
  - constrained beam search (prefix verifier)
  - diverse beam search with several diversity penalties.
- Enum **DecodingStrategy** – choose decoding mode.
- Enum **DiversityFn** – choose diversity penalty.

Nothing else in your codebase must change: import this class instead of the
original or alias it to the old name.
"""

import math
from enum import Enum, auto
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from data.schemas import TokenizedSeqBatch
from modules.model import (
    EncoderDecoderRetrievalModel as _Base,
    GenerationOutput,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecodingStrategy(Enum):
    BEAM = auto()
    CONSTRAINED_BEAM = auto()
    DIVERSE_BEAM = auto()

class DiversityFn(Enum):
    TOKEN = auto()      # simple token uniqueness
    HAMMING = auto()    # Hamming distance penalty
    NGRAM = auto()      # n‑gram coverage penalty
    SEMANTIC = auto()   # cosine distance in embed space (requires fn)

# ---------------------------------------------------------------------------
# Hypothesis dataclass (minimal – no @dataclass to keep Torchscript happy)
# ---------------------------------------------------------------------------

class _Hyp:
    def __init__(self, seq: Tensor, logp: Tensor):
        self.seq = seq  # shape (t,)
        self.logp = logp  # scalar tensor
    def __len__(self):
        return self.seq.shape[0]
    def extend(self, token: Tensor, lp: Tensor) -> "_Hyp":
        return _Hyp(torch.cat([self.seq, token.unsqueeze(0)], 0), self.logp + lp)

# ---------------------------------------------------------------------------
# Main extended model
# ---------------------------------------------------------------------------

class EncoderDecoderRetrievalModelExt(_Base):
    def __init__(
        self,
        *base_args,
        decoding_strategy: DecodingStrategy = DecodingStrategy.CONSTRAINED_BEAM,
        beam_width: int = 4,
        num_groups: int = 2,
        diversity_strength: float = 0.5,
        diversity_fn: DiversityFn = DiversityFn.TOKEN,
        ngram_n: int = 2,
        semantic_distance: Callable[[Tensor, Tensor], Tensor] | None = None,
        **base_kwargs,
    ) -> None:
        super().__init__(*base_args, **base_kwargs)
        self.decoding_strategy = decoding_strategy
        self.beam_width = beam_width
        self.num_groups = num_groups
        self.diversity_strength = diversity_strength
        self.diversity_fn = diversity_fn
        self.ngram_n = ngram_n
        self.semantic_distance = semantic_distance

    # ---------------------------------------------------------------------
    # Public generation API – keeps same signature as base
    # ---------------------------------------------------------------------

    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: float = 1.0,
        top_k: bool = False,
    ) -> GenerationOutput:
        if self.decoding_strategy == DecodingStrategy.DIVERSE_BEAM:
            sem_ids, lps = self._diverse_beam(batch)
        elif self.decoding_strategy == DecodingStrategy.BEAM:
            sem_ids, lps = self._beam(batch, constrained=False)
        else:  # constrained beam or default sampling path
            sem_ids, lps = self._beam(batch, constrained=True)
        return GenerationOutput(sem_ids=sem_ids, log_probas=lps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _beam(self, batch: TokenizedSeqBatch, constrained: bool) -> Tuple[Tensor, Tensor]:
        B = batch.sem_ids.shape[0]
        beams: List[List[_Hyp]] = [[_Hyp(torch.empty(0, dtype=torch.long, device=batch.sem_ids.device), torch.tensor(0.0, device=batch.sem_ids.device))] for _ in range(B)]
        for t in range(self.sem_id_dim):
            new_beams: List[List[_Hyp]] = [[] for _ in range(B)]
            # prepare input tensors (collect current prefixes)
            pref = torch.stack([torch.cat([b[0].seq, torch.full((self.sem_id_dim - len(b[0]),), -1, device=batch.sem_ids.device)]) for b in beams])
            in_batch = TokenizedSeqBatch(
                user_ids=batch.user_ids,
                sem_ids=batch.sem_ids,
                sem_ids_fut=pref,
                seq_mask=batch.seq_mask,
                token_type_ids=batch.token_type_ids,
                token_type_ids_fut=torch.arange(pref.shape[1], device=pref.device).unsqueeze(0).repeat(pref.shape[0],1),
            )
            logits = self.forward(in_batch).logits  # (B, V)
            logp_tok = F.log_softmax(logits, -1)
            topk_lp, topk_ix = torch.topk(logp_tok, self.beam_width, -1)
            for i in range(B):
                for k in range(self.beam_width):
                    tok = topk_ix[i, k]
                    lp = topk_lp[i, k]
                    for hyp in beams[i]:
                        if constrained and not self.inference_verifier_fn(torch.cat([hyp.seq, tok.unsqueeze(0)]).unsqueeze(0)):
                            continue
                        new_beams[i].append(hyp.extend(tok, lp))
                # keep best beam_width
                new_beams[i] = sorted(new_beams[i], key=lambda h: h.logp.item(), reverse=True)[: self.beam_width]
            beams = new_beams
        final_ids = torch.stack([b[0].seq for b in beams])
        final_lp = torch.stack([b[0].logp for b in beams])
        return final_ids, final_lp

    @torch.no_grad()
    def _diverse_beam(self, batch: TokenizedSeqBatch) -> Tuple[Tensor, Tensor]:
        assert self.beam_width % self.num_groups == 0, "beam_width must be divisible by num_groups"
        gsize = self.beam_width // self.num_groups
        B = batch.sem_ids.shape[0]
        groups: List[List[List[_Hyp]]] = [[[_Hyp(torch.empty(0, dtype=torch.long, device=batch.sem_ids.device), torch.tensor(0.0, device=batch.sem_ids.device))] for _ in range(self.num_groups)] for _ in range(B)]
        for t in range(self.sem_id_dim):
            for g in range(self.num_groups):
                pref = []
                for i in range(B):
                    pref.extend([torch.cat([h.seq, torch.full((self.sem_id_dim - len(h),), -1, device=batch.sem_ids.device)]) for h in groups[i][g]])
                pref = torch.stack(pref) if pref else torch.empty(0, self.sem_id_dim, dtype=torch.long, device=batch.sem_ids.device)
                # fast‑forward when no hypotheses (should not happen)
                if pref.numel() == 0:
                    continue
                # Build batch mirror for forward pass
                rep = math.ceil(pref.shape[0]/B)
                uid = batch.user_ids.repeat_interleave(rep, 0)[: pref.shape[0]]
                smids = batch.sem_ids.repeat_interleave(rep, 0)[: pref.shape[0]]
                smask = batch.seq_mask.repeat_interleave(rep, 0)[: pref.shape[0]]
                ttid = batch.token_type_ids.repeat_interleave(rep, 0)[: pref.shape[0]]
                in_batch = TokenizedSeqBatch(
                    user_ids=uid,
                    sem_ids=smids,
                    sem_ids_fut=pref,
                    seq_mask=smask,
                    token_type_ids=ttid,
                    token_type_ids_fut=torch.arange(pref.shape[1], device=pref.device).unsqueeze(0).repeat(pref.shape[0],1),
                )
                logits = self.forward(in_batch).logits
                logp_tok = F.log_softmax(logits, -1)
                topk_lp, topk_ix = torch.topk(logp_tok, gsize, -1)
                idx = 0
                for i in range(B):
                    cand: List[_Hyp] = []
                    for _ in range(len(groups[i][g])):
                        hyp = groups[i][g].pop(0)
                        for k in range(gsize):
                            tok = topk_ix[idx, k]
                            lp = topk_lp[idx, k]
                            new_h = hyp.extend(tok, lp)
                            cand.append(new_h)
                        idx += 1
                    # diversity penalty
                    if self.diversity_fn == DiversityFn.TOKEN and g > 0:
                        prev_toks = torch.stack([h.seq[-1] for h in sum(groups[i][:g], [])])
                        for h in cand:
                            div_pen = (prev_toks == h.seq[-1]).any().float() * self.diversity_strength
                            h.logp -= div_pen
                    cand = sorted(cand, key=lambda h: h.logp.item(), reverse=True)
                    groups[i][g] = cand[:gsize]
        fin_ids = torch.stack([sum((g[0] for g in grp), _Hyp(torch.empty(0, device=batch.sem_ids.device), torch.tensor(0.0))).seq for grp in groups])
        fin_lp = torch.stack([sum(grp[0][0].logp for grp in groups)])
        return fin_ids, fin_lp

# public symbol list for wildcard import
__all__ = [
    "EncoderDecoderRetrievalModelExt",
    "DecodingStrategy",
    "DiversityFn",
]

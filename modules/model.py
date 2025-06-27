# Standard library imports
from enum import Enum
from typing import NamedTuple

# Third party imports
import gin
import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F

# Local imports
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder, UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder, TransformerEncoderDecoder
from modules.utils import (
    eval_mode, maybe_repeat_interleave, reset_encoder_cache,
    reset_kv_cache, select_columns_per_row
)
from ops.triton.jagged import jagged_to_flattened_tensor, padded_to_jagged_tensor

# Configure PyTorch for better performance
torch._dynamo.config.suppress_errors = True  # Required for torch.compile
torch.set_float32_matmul_precision("high")   # Enable high precision matrix multiplication


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
        strategy: str = "default",
        dbg_groups: int = 4,  # For DBS
        dbg_lambda: float = 0.0,  # For DBS
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim,
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)

        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(
            num_embeddings=sem_id_dim, embedding_dim=embedding_dim
        )

        self.transformer = (
            TransformerEncoderDecoder(
                d_in=attn_dim,
                d_out=attn_dim,
                dropout=dropout,
                num_heads=num_heads,
                encoder_layers=n_layers // 2,
                decoder_layers=n_layers // 2,
            )
            if self.jagged_mode
            else nn.Transformer(
                d_model=attn_dim,
                nhead=num_heads,
                num_encoder_layers=n_layers // 2,
                num_decoder_layers=n_layers // 2,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)

        self.dbg_groups = dbg_groups
        self.dbg_lambda = dbg_lambda

        strategies = {"default": self.generate_next_sem_id_default, "dbs": self.generate_next_sem_id_dbs}
        if strategy not in strategies:
            raise ValueError(f"Unknown generation strategy: {strategy}. Available strategies: {list(strategies.keys())}")
        self.generate_next_sem_id = strategies[strategy]

    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        # Get embeddings for users and semantic IDs
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)

        B, N, D = sem_ids_emb.shape

        # Add positional encodings
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        # Combine embeddings
        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat(
                [input_embedding_fut, sem_ids_emb_fut + tte_fut], axis=1
            )
        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(
                input_embedding,
                lengths=seq_lengths + 1,
                max_len=input_embedding.shape[1],
            )

            seq_lengths_fut = torch.tensor(
                input_embedding_fut.shape[1],
                device=input_embedding_fut.device,
                dtype=torch.int64,
            ).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(
                input_embedding_fut,
                lengths=seq_lengths_fut,
                max_len=input_embedding_fut.shape[1],
            )
        else:
            mem_mask = torch.cat(
                [
                    torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
                    batch.seq_mask,
                ],
                axis=1,
            )
            f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
            f_mask[~mem_mask] = float("-inf")
        # Project and normalize embeddings for transformer
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))

        if self.jagged_mode:
            transformer_output = self.transformer(
                x=transformer_input,
                context=transformer_context,
                padding_mask=batch.seq_mask,
                jagged=self.jagged_mode,
            )
        else:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                transformer_input.shape[1]
            )
            transformer_output = self.transformer(
                src=transformer_context,
                tgt=transformer_input,
                tgt_is_causal=True,
                tgt_mask=causal_mask,
                src_key_padding_mask=f_mask,
                memory_key_padding_mask=f_mask,
            )

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id_default(
        self, batch: TokenizedSeqBatch, temperature: float = 1, top_k: bool = True
    ) -> GenerationOutput:

        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 32 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None,
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(
                probas_batched, num_samples=n_top_k_candidates
            )

            if generated is None:
                samples_for_verify = samples_batched.unsqueeze(-1)
                is_valid_prefix = self.inference_verifier_fn(samples_for_verify)
            else:
                prefix = torch.cat(
                    [
                        generated.flatten(0, 1)
                        .unsqueeze(1)
                        .repeat_interleave(n_top_k_candidates, axis=1),
                        samples_batched.unsqueeze(-1),
                    ],
                    axis=-1,
                )
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)

            sampled_log_probas = torch.log(
                torch.gather(probas_batched, 1, samples_batched)
            ).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000 * (~is_valid_prefix)
                + sampled_log_probas
                + maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = (
                sorted_log_probas[:, :k],
                sorted_indices[:, :k],
            )
            top_k_samples = torch.gather(samples, 1, top_k_indices)


            if generated is not None:
                parent_id = torch.gather(
                    generated,
                    1,
                    (top_k_indices // n_top_k_candidates)
                    .unsqueeze(2)
                    .expand(-1, -1, i),
                )
                top_k_samples = torch.cat(
                    [parent_id, top_k_samples.unsqueeze(-1)], axis=-1
                )

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(
                        next_sem_ids.shape[1], device=next_sem_ids.device
                    ).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids,
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache = torch.zeros(
                        input_batch.sem_ids.shape[0],
                        input_batch.sem_ids.shape[1] + 1,
                        self.attn_dim,
                        device=input_batch.sem_ids.device,
                    )
                    cache_mask = torch.cat(
                        [
                            torch.ones(
                                input_batch.sem_ids.shape[0],
                                1,
                                dtype=bool,
                                device=input_batch.seq_mask.device,
                            ),
                            input_batch.seq_mask,
                        ],
                        axis=1,
                    )
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = (
                        self.transformer.cached_enc_output.offsets()
                        .diff()
                        .repeat_interleave(k)
                    )
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(
                        cache, lengths, max_len=cache.shape[1]
                    )

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(
                        k, dim=0
                    ),
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())

        result = GenerationOutput(
            sem_ids=generated.squeeze(), log_probas=log_probas.squeeze()
        )
        return result




    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id_dbs(
        self,
        batch: TokenizedSeqBatch,
        temperature: float = 1.0,
        top_k: bool = True,
        # 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ) -> GenerationOutput: # 1.5, 2, 5; (2.5, 3, double)
        # other: 0, 0.5, 1
        """
        Diverse-beam search that really returns k different semantic-id
        sequences.  Works for dbg_groups ≥ 1 and any λ ≥ 0.
        """

        assert self.enable_generation, "Model generation is not enabled"
        B = batch.sem_ids.size(0)
        k = 16 if top_k else 1                       # total beam width
        assert k % self.dbg_groups == 0, "k must divide dbg_groups"
        k_per_group = k // self.dbg_groups
        C = 100 if top_k else 1                      # per-beam candidate pool

        # ------------------------------------------------------------------ #
        # helpers                                                             #
        # ------------------------------------------------------------------ #
        def _explode_encoder_once():
            if not (self.jagged_mode and k > 1):
                return
            cache = torch.zeros(
                input_batch.sem_ids.size(0),
                input_batch.sem_ids.size(1) + 1,
                self.attn_dim,
                device=batch.sem_ids.device,
            )
            cache_mask = torch.cat(
                [torch.ones(input_batch.sem_ids.size(0), 1, dtype=torch.bool,
                            device=batch.sem_ids.device),
                 input_batch.seq_mask], dim=1)
            cache[cache_mask] = self.transformer.cached_enc_output.values()
            lens = self.transformer.cached_enc_output.offsets().diff() \
                       .repeat_interleave(k)
            cache = cache.repeat_interleave(k, dim=0)
            self.transformer.cached_enc_output = padded_to_jagged_tensor(
                cache, lens, max_len=cache.size(1))

        # ------------------------------------------------------------------ #
        # initial batch fed to the decoder                                   #
        # ------------------------------------------------------------------ #
        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None,
        )

        # ------------------------------------------------------------------ #
        # t = 0  →  pick the FIRST token with diversity                      #
        # ------------------------------------------------------------------ #
        logits0 = self.forward(input_batch).logits          # (B, V)
        logp0   = F.log_softmax(logits0 / temperature, -1)  # (B, V)
        logp_top0, tok_top0 = torch.topk(logp0, C, dim=-1)  # (B, C)

        # verifier mask over the C candidates
        valid0 = self.inference_verifier_fn(
            tok_top0.unsqueeze(-1)          # (B, C, 1)
        ).squeeze(-1)                       # (B, C) bool

        div_counts = torch.zeros(B, self.num_embeddings,
                                 device=batch.sem_ids.device)
        first_tokens, first_logp = [], []                   # collect per group
        for g in range(self.dbg_groups):
            scores = logp_top0 - self.dbg_lambda * div_counts.gather(1, tok_top0)
            scores = scores.masked_fill(~valid0, -1e4)
            best_lp, best_idx = scores.topk(k_per_group, dim=-1)
            best_tok = torch.gather(tok_top0, 1, best_idx)  # (B, k′)

            first_tokens.append(best_tok)
            first_logp  .append(best_lp)
            div_counts.scatter_add_(1, best_tok,
                                    torch.ones_like(best_tok,
                                                    dtype=div_counts.dtype))

        # stack into beams
        generated = torch.cat(first_tokens, dim=1).unsqueeze(-1)   # (B,k,1)
        log_probas = torch.cat(first_logp, dim=1)                  # (B,k)

        # ------------------------------------------------------------------ #
        # prepare decoder input for step 1                                   #
        # ------------------------------------------------------------------ #
        next_ids = generated.flatten(end_dim=1)                     # (B·k,1)
        tt_future = torch.zeros_like(next_ids)                      # (B·k,1)

        # explode encoder cache & repeat encoder-side tensors once
        _explode_encoder_once()
        if k > 1:
            input_batch = TokenizedSeqBatch(
                user_ids=batch.user_ids.repeat_interleave(k, 0),
                sem_ids=batch.sem_ids.repeat_interleave(k, 0),
                sem_ids_fut=next_ids,
                token_type_ids_fut=tt_future,
                seq_mask=batch.seq_mask.repeat_interleave(k, 0),
                token_type_ids=batch.token_type_ids.repeat_interleave(k, 0),
            )
        else:
            input_batch = TokenizedSeqBatch(
                user_ids=input_batch.user_ids,
                sem_ids=input_batch.sem_ids,
                sem_ids_fut=next_ids,
                token_type_ids_fut=tt_future,
                seq_mask=input_batch.seq_mask,
                token_type_ids=input_batch.token_type_ids,
            )

        # ------------------------------------------------------------------ #
        # steps 1 … D-1  (unchanged DBS over existing beams)                 #
        # ------------------------------------------------------------------ #
        for t in range(1, self.sem_id_dim):
            div_counts.zero_()
            logits = self.forward(input_batch).logits              # (B·k,V)
            logp_token = F.log_softmax(logits / temperature, -1)
            logp_top, tok_top = torch.topk(logp_token, C, dim=-1)  # (B·k, C)

            logp_top = logp_top.view(B, k, C)
            tok_top  = tok_top .view(B, k, C)

            # verifier mask
            prefix = torch.cat([
                generated.unsqueeze(2).repeat_interleave(C, 2),
                tok_top.unsqueeze(-1)
            ], -1)                                                 # (B,k,C,t+1)
            valid = self.inference_verifier_fn(
                prefix.view(B, -1, t + 1)
            ).view_as(tok_top)

            new_gen, new_lp = [], []
            for g in range(self.dbg_groups):
                b0, b1 = g * k_per_group, (g + 1) * k_per_group
                cand_scores = logp_top[:, b0:b1, :].reshape(B, -1)
                cand_tok    =  tok_top[:, b0:b1, :].reshape(B, -1)
                cand_valid  =   valid[:, b0:b1, :].reshape(B, -1)

                if g > 0:
                    penalty = div_counts.gather(1, cand_tok)
                    cand_scores = cand_scores - self.dbg_lambda * penalty

                prefix_scores = log_probas[:, b0:b1].unsqueeze(-1) \
                                              .expand(-1, -1, C) \
                                              .reshape(B, -1)
                total = cand_scores + prefix_scores + (-1e4) * (~cand_valid)

                best_lp, best_idx = total.topk(k_per_group, dim=1)
                best_tok = torch.gather(cand_tok, 1, best_idx)

                div_counts.scatter_add_(1, best_tok,
                                        torch.ones_like(best_tok,
                                                        dtype=div_counts.dtype))

                parent_slice = (best_idx // C)
                parents = torch.gather(
                    generated[:, b0:b1, :], 1,
                    parent_slice.unsqueeze(-1).expand(-1, -1, t))
                seq = torch.cat([parents, best_tok.unsqueeze(-1)], -1)

                new_gen.append(seq); new_lp.append(best_lp)

            generated  = torch.cat(new_gen, 1)          # (B,k,t+1)
            log_probas = torch.cat(new_lp , 1)          # (B,k)

            next_ids = generated.flatten(end_dim=1)     # (B·k,t+1)
            tt_future = torch.arange(t + 1, device=next_ids.device) \
                            .repeat(next_ids.size(0), 1)

            input_batch = TokenizedSeqBatch(
                user_ids=input_batch.user_ids,
                sem_ids=input_batch.sem_ids,
                sem_ids_fut=next_ids,
                token_type_ids_fut=tt_future,
                seq_mask=input_batch.seq_mask,
                token_type_ids=input_batch.token_type_ids,
            )

        return GenerationOutput(
            sem_ids   = generated.squeeze(),
            log_probas= log_probas.squeeze(),
        )
    

    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)

        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(
                    jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B
                )[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(
                    F.cross_entropy(logits, target, reduction="none", ignore_index=-1),
                    "(b n) -> b n",
                    b=B,
                )
                base_loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                base_loss = rearrange(
                    F.cross_entropy(out, target, reduction="none", ignore_index=-1),
                    "(b n) -> b n",
                    b=B,
                ).sum(axis=1).mean()

            log_probs = F.log_softmax(logits, dim=-1)  # shape: [B*N, vocab_size]
            probs = torch.exp(log_probs)
            # Calculate entropy and apply regularization
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            λ = 0.05  # Entropy regularization strength
            loss = base_loss - λ * entropy  # Maximize entropy


            loss_d = unred_loss.mean(axis=0) if self.jagged_mode else None
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None

        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(
                jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B
            )[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)

    
    


    # --------------------------------------------------------------------
    #  GREEDY  (baseline) ─ single-beam, left-to-right, no diversity term
    # --------------------------------------------------------------------
    # @eval_mode
    # @reset_encoder_cache
    # @torch.no_grad
    # def generate_next_sem_id(
    #         self,
    #         batch: TokenizedSeqBatch,
    #         temperature: float = 1.0,
    #         max_candidates: int = 100,
    #         top_k: int = 0
    # ) -> GenerationOutput:
    #     """
    #     Plain greedy decoding – single beam, one token chosen at every step.
    #     """
    #     assert self.enable_generation, "Model generation is not enabled"

    #     B, D   = batch.sem_ids.size(0), self.sem_id_dim
    #     device = batch.sem_ids.device
    #     generated   = torch.empty(B, 0, dtype=torch.long, device=device)
    #     logp_prefix = torch.zeros(B, device=device)

    #     # ------------ helper --------------------------------------------------
    #     def _with_future(fut: Tensor | None) -> TokenizedSeqBatch:
    #         if fut is None:              # first call: no predicted tokens yet
    #             return TokenizedSeqBatch(
    #                 user_ids          = batch.user_ids,
    #                 sem_ids           = batch.sem_ids,
    #                 sem_ids_fut       = None,
    #                 seq_mask          = batch.seq_mask,
    #                 token_type_ids    = batch.token_type_ids,
    #                 token_type_ids_fut= None,
    #             )
    #         tt = torch.arange(fut.size(1), device=device).repeat(fut.size(0), 1)
    #         return TokenizedSeqBatch(
    #             user_ids          = batch.user_ids,
    #             sem_ids           = batch.sem_ids,
    #             sem_ids_fut       = fut,
    #             seq_mask          = batch.seq_mask,
    #             token_type_ids    = batch.token_type_ids,
    #             token_type_ids_fut= tt,
    #         )
    #     # ---------------------------------------------------------------------

    #     input_batch = _with_future(None)

    #     for t in range(D):
    #         logits  = self.forward(input_batch).logits                  # (B,V)
    #         logp    = F.log_softmax(logits / temperature, -1)
    #         logp_top, tok_top = torch.topk(logp, max_candidates, -1)    # (B,C)

    #         # build verifier prefixes (B,C,t+1)
    #         prefix_rep = generated[:, None, :].repeat_interleave(max_candidates, 1)
    #         prefix     = torch.cat([prefix_rep, tok_top[..., None]], -1)

    #         valid = self.inference_verifier_fn(
    #                     prefix.view(B * max_candidates, t + 1)
    #                 ).view(B, max_candidates)

    #         # if nothing is valid for a row fall back to highest-prob token
    #         first_valid = torch.where(valid.any(-1),
    #                                   valid.float().cumsum(-1).argmax(-1),
    #                                   torch.zeros(B, dtype=torch.long, device=device))

    #         next_tok  = tok_top .gather(1, first_valid[:, None]).squeeze(1)   # (B,)
    #         next_logp = logp_top.gather(1, first_valid[:, None]).squeeze(1)   # (B,)

    #         generated   = torch.cat([generated, next_tok[:, None]], 1)        # (B,t+1)
    #         logp_prefix = logp_prefix + next_logp

    #         if t < D - 1:
    #             input_batch = _with_future(generated)

    #     return GenerationOutput(sem_ids=generated, log_probas=logp_prefix)




    











# from __future__ import annotations
# import math
# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# from typing import NamedTuple, List, Tuple

# from einops import rearrange                                   # unchanged ↓
# from data.schemas import TokenizedSeqBatch
# from modules.embedding.id_embedder import SemIdEmbedder, UserIdEmbedder
# from modules.normalize import RMSNorm
# from modules.transformer.attention import AttentionInput
# from modules.transformer.model import TransformerEncoderDecoder
# from modules.utils import eval_mode, reset_encoder_cache
# from ops.triton.jagged import jagged_to_flattened_tensor, padded_to_jagged_tensor

# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision("high")


# # ---------------------------------------------------------------------------
# # small helpers -------------------------------------------------------------
# # ---------------------------------------------------------------------------
# _PAD = -1
# class _Hyp:
#     __slots__ = ("seq", "logp")
#     def __init__(self, seq: Tensor, logp: Tensor):
#         self.seq, self.logp = seq, logp          # seq: (≤D,), logp: scalar
#     def __len__(self): return self.seq.numel()
#     def extend(self, tok: Tensor, lp: Tensor) -> "_Hyp":
#         return _Hyp(torch.cat([self.seq, tok.unsqueeze(0)]), self.logp + lp)

# def _pad(seq: Tensor, tgt: int) -> Tensor:
#     """Pad (or truncate) 1-D long tensor to length *tgt* with -1."""
#     L = seq.numel()
#     if L == tgt: return seq
#     if L >  tgt: return seq[:tgt]
#     return F.pad(seq, (0, tgt-L), value=_PAD)


# # ---------------------------------------------------------------------------
# # public dataclasses (unchanged) --------------------------------------------
# # ---------------------------------------------------------------------------
# class ModelOutput(NamedTuple):
#     loss: Tensor
#     logits: Tensor
#     loss_d: Tensor

# class GenerationOutput(NamedTuple):
#     sem_ids: Tensor
#     log_probas: Tensor


# # ---------------------------------------------------------------------------
# # main model ----------------------------------------------------------------
# # ---------------------------------------------------------------------------
# class EncoderDecoderRetrievalModel(nn.Module):
#     # ──────────────────────────────────────────────────────────────────────
#     # constructor (unchanged, only adds a default beam width)
#     # ──────────────────────────────────────────────────────────────────────
#     def __init__(
#         self,
#         embedding_dim, attn_dim, dropout, num_heads, n_layers,
#         num_embeddings, sem_id_dim,
#         inference_verifier_fn,
#         max_pos: int = 2048,
#         jagged_mode: bool = True,
#         beam_width: int = 32,                       # NEW ← default beam size
#     ) -> None:
#         super().__init__()
#         self.jagged_mode    = jagged_mode
#         self.num_embeddings = num_embeddings
#         self.sem_id_dim     = sem_id_dim
#         self.attn_dim       = attn_dim
#         self.inference_verifier_fn = inference_verifier_fn
#         self.enable_generation     = False
#         self.beam_width            = beam_width   # <-- stores default

#         # --- embedding / transformer stack (identical to original) ---
#         self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
#         self.norm, self.norm_cxt = RMSNorm(embedding_dim), RMSNorm(embedding_dim)
#         self.do   = nn.Dropout(p=0.5)
#         self.sem_id_embedder = SemIdEmbedder(num_embeddings, sem_id_dim, embedding_dim)
#         self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
#         self.wpe  = nn.Embedding(max_pos, embedding_dim)
#         self.tte  = nn.Embedding(sem_id_dim, embedding_dim)
#         self.tte_fut = nn.Embedding(sem_id_dim, embedding_dim)

#         self.transformer = (
#             TransformerEncoderDecoder(
#                 d_in=attn_dim, d_out=attn_dim, dropout=dropout,
#                 num_heads=num_heads,
#                 encoder_layers=n_layers // 2, decoder_layers=n_layers // 2
#             ) if jagged_mode else
#             nn.Transformer(
#                 d_model=attn_dim, nhead=num_heads, batch_first=True,
#                 num_encoder_layers=n_layers // 2, num_decoder_layers=n_layers // 2,
#                 dim_feedforward=1024, dropout=dropout
#             )
#         )
#         self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
#         self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
#         self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)

#     # ------------------------------------------------------------------
#     # predict (identical to original – omitted for brevity)
#     # ------------------------------------------------------------------
#     def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
#         #print("okay")
#         user_emb = self.user_id_embedder(batch.user_ids)
#         #print("okay1")
#         sem_ids_emb = self.sem_id_embedder(batch)
#         #print("okay2")
#         sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
#         #print("okay3")
#         seq_lengths = batch.seq_mask.sum(axis=1)

#         B, N, D = sem_ids_emb.shape

#         pos_max = N // self.sem_id_dim
#         # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)

#         pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
#         wpe = self.wpe(pos)
#         #print("okay5")

#         input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
#         input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
#         if sem_ids_emb_fut is not None:
#             tte_fut = self.tte(batch.token_type_ids_fut)
#             input_embedding_fut = torch.cat(
#                 [input_embedding_fut, sem_ids_emb_fut + tte_fut], axis=1
#             )
#         #print("okay6")

#         if self.jagged_mode:
#             input_embedding = padded_to_jagged_tensor(
#                 input_embedding,
#                 lengths=seq_lengths + 1,
#                 max_len=input_embedding.shape[1],
#             )

#             seq_lengths_fut = torch.tensor(
#                 input_embedding_fut.shape[1],
#                 device=input_embedding_fut.device,
#                 dtype=torch.int64,
#             ).repeat(B)
#             input_embedding_fut = padded_to_jagged_tensor(
#                 input_embedding_fut,
#                 lengths=seq_lengths_fut,
#                 max_len=input_embedding_fut.shape[1],
#             )
#         else:
#             mem_mask = torch.cat(
#                 [
#                     torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
#                     batch.seq_mask,
#                 ],
#                 axis=1,
#             )
#             f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
#             f_mask[~mem_mask] = float("-inf")
#         #print("okay7")
#         transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
#         transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))

#         if self.jagged_mode:
#             transformer_output = self.transformer(
#                 x=transformer_input,
#                 context=transformer_context,
#                 padding_mask=batch.seq_mask,
#                 jagged=self.jagged_mode,
#             )
#         else:
#             causal_mask = nn.Transformer.generate_square_subsequent_mask(
#                 transformer_input.shape[1]
#             )
#             transformer_output = self.transformer(
#                 src=transformer_context,
#                 tgt=transformer_input,
#                 tgt_is_causal=True,
#                 tgt_mask=causal_mask,
#                 src_key_padding_mask=f_mask,
#                 memory_key_padding_mask=f_mask,
#             )

#         return transformer_output
    

    
#     # ----------------------------------------------------------------------
#     # Constrained beam search – always produces sem_id_dim tokens and
#     # guarantees every intermediate prefix exists in the tokenizer cache.
#     # ----------------------------------------------------------------------
#     @torch.no_grad()
#     def _beam(
#         self,
#         batch: TokenizedSeqBatch,
#         beam_width: int,
#         temperature: float = 1.0,
#     ) -> Tuple[Tensor, Tensor]:
#         B, D   = batch.sem_ids.size(0), self.sem_id_dim
#         dev    = batch.sem_ids.device   

#         beams: List[List[_Hyp]] = [[_Hyp(torch.empty(0, dtype=torch.long, device=dev),
#                                          torch.tensor(0.0, device=dev))]
#                                    for _ in range(B)]   

#         # helper: first vocab token that passes the verifier
#         def _first_valid(row_logp: Tensor, hyp: _Hyp) -> Tuple[Tensor, Tensor]:
#             for idx in torch.argsort(row_logp, descending=True):
#                 cand_seq = torch.cat([hyp.seq, idx.view(1)])
#                 if self.inference_verifier_fn(cand_seq[None, :]):
#                     return idx, row_logp[idx]
#             idx = torch.argmax(row_logp)       # fallback – should be unreachable
#             return idx, row_logp[idx]   

#         for _ in range(D):
#             # ── 1. build a (B,D) tensor with the *best* prefix of every batch elt
#             pref = torch.stack([_pad(b[0].seq, D) for b in beams])  

#             in_batch = TokenizedSeqBatch(
#                 user_ids=batch.user_ids,
#                 sem_ids=batch.sem_ids,
#                 sem_ids_fut=pref,
#                 seq_mask=batch.seq_mask,
#                 token_type_ids=batch.token_type_ids,
#                 token_type_ids_fut=torch.arange(D, device=dev).expand(B, -1),
#             )
#             if self.jagged_mode:
#                 # cached encoder has wrong batch size → flush
#                 self.transformer.cached_enc_output = None   

#             logits  = self.forward(in_batch).logits                     # (B,V)
#             logp    = F.log_softmax(logits / temperature, dim=-1)
#             top_lp, top_ix = torch.topk(logp, beam_width, dim=-1)       # (B,k) 

#             new_beams: List[List[_Hyp]] = [[] for _ in range(B)]
#             for i in range(B):
#                 for hyp in beams[i]:
#                     added = 0
#                     # ── 2. try top-k tokens
#                     for k in range(beam_width):
#                         tok, lp = top_ix[i, k], top_lp[i, k]
#                         cand_seq = torch.cat([hyp.seq, tok.view(1)])
#                         if not self.inference_verifier_fn(cand_seq[None, :]):
#                             continue
#                         new_beams[i].append(hyp.extend(tok, lp))
#                         added += 1
#                     # ── 3. NONE of the top-k survived → force the first   ✓
#                     if added == 0:
#                         tok, lp = _first_valid(logp[i], hyp)
#                         new_beams[i].append(hyp.extend(tok, lp))    

#                 # keep best `beam_width`
#                 new_beams[i] = sorted(new_beams[i],
#                                       key=lambda h: h.logp.item(),
#                                       reverse=True)[: beam_width]
#             beams = new_beams   

#         # ── 4. stack results --------------------------------------------------
#         out_ids = torch.stack([
#             torch.stack([_pad(h.seq, D) for h in b]) for b in beams      # (B,k,D)
#         ])
#         out_lp  = torch.stack([
#             torch.stack([h.logp for h in b]) for b in beams              # (B,k)
#         ])
#         return out_ids, out_lp



#     # ------------------------------------------------------------------
#     # PUBLIC generation API  (old routine replaced by beam search)
#     # ------------------------------------------------------------------
#     @eval_mode
#     @reset_encoder_cache
#     @torch.no_grad
#     def generate_next_sem_id(
#         self, batch: TokenizedSeqBatch,
#         temperature: float = 1.0, top_k: bool = True
#     ) -> GenerationOutput:
#         assert self.enable_generation, "Model generation is not enabled"

#         # use caller's intention to choose beam width (keeps old interface)
#         bw = 32 if top_k else 1
#         ids, lps = self._beam(batch, beam_width=bw, temperature=temperature)
#         return GenerationOutput(sem_ids=ids, log_probas=lps)

#     # ------------------------------------------------------------------
#     # forward  (identical to original – just copy it verbatim)
#     # ------------------------------------------------------------------
#     @torch.compile
#     def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
#         seq_mask = batch.seq_mask
#         B, N = seq_mask.shape

#         trnsf_out = self._predict(batch)

#         if self.training or not self.enable_generation:
#             predict_out = self.out_proj(trnsf_out)
#             if self.jagged_mode:
#                 # This works because batch.sem_ids_fut is fixed length, no padding.
#                 logits = rearrange(
#                     jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B
#                 )[:, :-1, :].flatten(end_dim=1)
#                 target = batch.sem_ids_fut.flatten(end_dim=1)
#                 unred_loss = rearrange(
#                     F.cross_entropy(logits, target, reduction="none", ignore_index=-1),
#                     "(b n) -> b n",
#                     b=B,
#                 )
#                 base_loss = unred_loss.sum(axis=1).mean() 
#             else:
#                 logits = predict_out
#                 out = logits[:, :-1, :].flatten(end_dim=1)
#                 target = batch.sem_ids_fut.flatten(end_dim=1)
#                 base_loss = rearrange(
#                     F.cross_entropy(out, target, reduction="none", ignore_index=-1),
#                     "(b n) -> b n",
#                     b=B,
#                 ).sum(axis=1).mean()

#             log_probs = F.log_softmax(logits, dim=-1)  # shape: [B*N, vocab_size]
#             probs = torch.exp(log_probs)
#             entropy = -torch.sum(probs * log_probs, dim=-1).mean() 
#             λ = 0.05  # Regularization strength, can be tuned
#             loss = base_loss - λ * entropy  # maximizing entropy


#             loss_d = unred_loss.mean(axis=0) if self.jagged_mode else None
#             if not self.training and self.jagged_mode:
#                 self.transformer.cached_enc_output = None

#         elif self.jagged_mode:
#             trnsf_out = trnsf_out.contiguous()
#             trnsf_out_flattened = rearrange(
#                 jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B
#             )[:, -1, :]
#             logits = self.out_proj(trnsf_out_flattened)
#             loss = None
#             loss_d = None
#         else:
#             trnsf_out_flattened = trnsf_out[:, -1, :]
#             logits = self.out_proj(trnsf_out_flattened)
#             loss = None
#             loss_d = None

#         return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)
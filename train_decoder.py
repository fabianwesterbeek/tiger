import argparse
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
# from modules.model_ext import EncoderDecoderRetrievalModelExt as EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.tokenizer.lookup_table import SemanticIDLookupTable
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    category=None,
):
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")
    print(f"Starting training with dataset: {dataset}, split: {dataset_split}, category: {category}")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    print(f"Initialized Accelerator with split_batches={split_batches} and mixed_precision={mixed_precision_type if amp else 'no'}")

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(project="gen-retrieval-decoder-training", config=params)

    item_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
            split=dataset_split,
        )
        if category is None
        else ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
            split=dataset_split,
            category=category,
        )
    )
    print("Item dataset initialized.")

    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=True,
        subsample=train_data_subsample,
        split=dataset_split,
    )
    print("Train dataset initialized.")
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        split=dataset_split,
    )
    

    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train= False,
        is_eval_split="eval",
        subsample=False,
        split=dataset_split,
    )
    print("Evaluation dataset initialized.")


    test_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train= False,
        is_eval_split="test",
        subsample=False,
        split=dataset_split,
    )

    print("Test dataset initialized")
    


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
    train_dataloader, eval_dataloader, test_dataloader)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    print("Tokenizer initialized.")
    tokenizer = accelerator.prepare(tokenizer)
    print("DEBUG: Precomputing corpus IDs...")
    tokenizer.precompute_corpus_ids(item_dataset)
    print("DEBUG: Finished precomputing corpus IDs")

    # Create and build lookup table for ILD calculation
    if accelerator.is_main_process:
        print("Building semantic ID to embedding lookup table...")
        lookup_table = SemanticIDLookupTable(tokenizer.rq_vae)
        num_entries = lookup_table.build_lookup_table(item_dataset)
        print(f"Lookup table built with {num_entries} entries")

        # # Debug: Check if lookup table was built properly
        # sample_item = item_dataset[0]
        # sample_item_tensor = batch_to(sample_item, device).x
        # print(f"DEBUG: Sample item tensor shape: {sample_item_tensor.shape}")
        # with torch.no_grad():
        #     sem_ids = tokenizer.rq_vae.get_semantic_ids(sample_item_tensor).sem_ids
        #     print(f"DEBUG: Sample semantic ID shape: {sem_ids.shape}, value: {sem_ids.tolist()}")
        #     sem_id_tuple = tuple(sem_ids[0].cpu().tolist())
        #     print(f"DEBUG: Sample semantic ID tuple: {sem_id_tuple}")
        #     print(f"DEBUG: Is sample ID in lookup table: {sem_id_tuple in lookup_table.id_to_embedding_map}")
    else:
        lookup_table = None

    # -- some debugging --
    ## Debug information
    print(f"Dataset split: {dataset_split}")
    print(f"Max sequence length: {train_dataset.max_seq_len}")
    print(f"Semantic IDs dimension: {tokenizer.sem_ids_dim}")
    print(f"Maximum position encoding: {train_dataset.max_seq_len * tokenizer.sem_ids_dim}")

    # Add a sanity check
    assert train_dataset.max_seq_len * tokenizer.sem_ids_dim < 1025, (
        f"Position encoding exceeds maximum allowed value. "
        f"max_seq_len={train_dataset.max_seq_len}, "
        f"sem_ids_dim={tokenizer.sem_ids_dim}, "
        f"product={train_dataset.max_seq_len * tokenizer.sem_ids_dim}"
    )

    # -- end of debugging --

    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len * tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode,
    )

    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=10000)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)


    current_best = 0 
    patience = 5
    patience_counter = 0

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    print("Starting training loop...")
    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
        mininterval=1.0,
    ) as pbar:
        for iter in range(iterations):
            model.train()
            # print(f"Training iteration {iter + 1}/{iterations}")
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                #with accelerator.autocast():
                model_output = model(tokenized_data)
                loss = model_output.loss / gradient_accumulate_every
                total_loss += loss

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(
                        tokenized_data, model_output
                    )

                accelerator.backward(total_loss)
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f"loss: {total_loss.item():.4f}")

            accelerator.wait_for_everyone()

            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()


            if (iter + 1) % partial_eval_every == 0:
                model.eval()
                model.enable_generation = False
                print(f"Performing partial evaluation at iteration {iter + 1}")
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)

                    if wandb_logging and accelerator.is_main_process:
                        eval_debug_metrics = compute_debug_metrics(
                            tokenized_data, model_output_eval, "eval"
                        )
                        eval_debug_metrics["eval_loss"] = (
                            model_output_eval.loss.detach().cpu().item()
                        )
                        wandb.log(eval_debug_metrics)

            if (iter + 1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                print(f"Performing full evaluation at iteration {iter + 1}")
                with tqdm(
                    eval_dataloader,
                    desc=f"Eval {iter+1}",
                    disable=not accelerator.is_main_process,
                    mininterval=1.0,
                ) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        generated = model.generate_next_sem_id(
                            tokenized_data, top_k=True, temperature=1
                        )
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                        # add the tokenizer and lookup table for ILD calculation
                        metrics_accumulator.accumulate(
                            actual=actual, top_k=top_k, tokenizer=tokenizer, lookup_table=lookup_table
                        )
                eval_metrics = metrics_accumulator.reduce()

                print(eval_metrics)

                patience_counter +=1

                ## We can change the Early stopping metric
                if eval_metrics["ndcg@5"] > current_best:
                    current_best = eval_metrics["ndcg@5"]
                    patience_counter = 0
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{dataset_split}_best.pt")
                    print(f"Saving Current Best Model")

                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)

                metrics_accumulator.reset()

                if patience_counter > patience:
                    print("Early stopping. Done training")
                    print(f"Best Eval NDCG@5:{current_best}")
                    break


            if accelerator.is_main_process:
                if  iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{dataset_split}_final.pt")
                    print(f"Final Model Checkpoint Saved")

                if wandb_logging:
                    wandb.log(
                        {
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": total_loss.cpu().item(),
                            **train_debug_metrics,
                        }
                    )


            pbar.update(1)

    # best_checkpoint_path = pretrained_decoder_path
    # state = torch.load(best_checkpoint_path, map_location=device)

    model.load_state_dict(state["model"])
    model.eval()
    model.enable_generation = True
    metrics_accumulator.reset()

    with tqdm(
        test_dataloader,
        desc="Test",
        disable=not accelerator.is_main_process,
        mininterval=1.0,
        ) as pbar_test:
        for batch in pbar_test:

            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)

            generated = model.generate_next_sem_id(
            tokenized_data, top_k=True, temperature=1
            )
            actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
            # add the tokinzer
            print("pred[0]", generated.sem_ids[0, 0])   # almost certainly [-1 … -1]
            print("gold[0]", tokenized_data.sem_ids_fut[0])
            valid = tokenizer.exists_prefix(generated.sem_ids[0, 0].unsqueeze(0))
            print("generated prefix passes verifier ?", valid.item())
            metrics_accumulator.accumulate(
                actual=actual, top_k=top_k, tokenizer=tokenizer
            )

        test_metrics = metrics_accumulator.reduce()
        print("Final Test Metrics: ")
        print(test_metrics)


    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()


    # Always perform a full evaluation after training completes
#     model.eval()
#     model.enable_generation = True
#     print(f"Performing final full evaluation after training")
#     with tqdm(
#         eval_dataloader,
#         desc=f"Final Evaluation",
#         disable=not accelerator.is_main_process,
#         mininterval=1.0,
#     ) as pbar_eval:
#         for batch in pbar_eval:
#             data = batch_to(batch, device)
#             tokenized_data = tokenizer(data)

#             generated = model.generate_next_sem_id(
#                 tokenized_data, top_k=True, temperature=1
#             )
#             actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
#             # add the tokenizer and lookup table for ILD calculation
#             metrics_accumulator.accumulate(
#                 actual=actual, top_k=top_k, tokenizer=tokenizer, lookup_table=lookup_table
#             )
#     final_eval_metrics = metrics_accumulator.reduce()

#     print(final_eval_metrics)
#     if accelerator.is_main_process and wandb_logging:
#         wandb.log({**final_eval_metrics, "final_evaluation": True})

#     metrics_accumulator.reset()

#     if wandb_logging:
#         wandb.finish()


# if __name__ == "__main__":
#     parse_config()
#     train()
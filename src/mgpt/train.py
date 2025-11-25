import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from mgpt.gpt_download import download_and_load_gpt2

class train_model:
    """Class to handle training and evaluation of the Mini AstroGPT model
    """
    def __init__(self, model,
                train_data_loader,
                val_data_loader, 
                optimizer,
                device,
                num_epochs, 
                eval_iter,
                eval_freq,
                start_context,
                max_new_tokens,
                num_batches,
                tokenizer_name='gpt2',
                 *args, **kwargs):

        self.model = model.to(device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.eval_iter = eval_iter
        self.eval_freq = eval_freq
        self.start_context = start_context
        self.max_new_tokens = max_new_tokens
        self.tokenizer_name = tokenizer_name
        
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)        
        
    def loss_data_batch(self, input_batch, 
                        target_batch):
        if isinstance(input_batch, list):
            input_batch = torch.tensor(input_batch)
        if isinstance(target_batch, list):
            target_batch = torch.tensor(target_batch)

        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        if input_batch.dim() == 1:
            input_batch = input_batch.unsqueeze(0)         
        if target_batch.dim() == 1:
            target_batch = target_batch.unsqueeze(0)       

        input_batch = input_batch.long()
        target_batch = target_batch.long()
        
        logits = self.model(input_batch)
        loss = F.cross_entropy(logits.flatten(0, 1), 
                                target_batch.flatten())
        return loss
        
    def loss_data_loader(self, data_loader, num_batches):
        
        total_loss = 0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            return len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i < num_batches:
                    loss = self.loss_data_batch(input_batch, target_batch)
                    total_loss += loss.item()
                else:
                    break
        return total_loss/num_batches
        
        
    def train_model_basic(self):
        
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            for input_batch, target_batch in self.train_data_loader:
                self.optimizer.zero_grad()
                loss = self.loss_data_batch(input_batch, target_batch)
                
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % self.eval_freq == 0:
                    train_loss = float(loss.detach().item())
                    val_loss = self.eval_model()

                    # log per step (append scalar)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "f"Val loss {val_loss:.3f}")
            self.generate_and_print_sample()
        return train_losses, val_losses, track_tokens_seen
        
    def eval_model(self):
        self.model.eval()
        with torch.no_grad():
            val_losses = self.loss_data_loader(self.val_data_loader, self.eval_iter)
        self.model.train()
        return val_losses
    
    def generate_and_print_sample(self):
        self.model.eval()
        
        context_size = self.model.embed_layer.position_embedding.num_embeddings
        # input_ids = self.tokenizer.encode(self.start_context, allowed_special=set())

        ids = self.tokenizer.encode(self.start_context, allowed_special=set())
        input_ids = torch.tensor(ids, device=self.device).unsqueeze(0)


        if input_ids.size(1) > context_size:
            input_ids = input_ids[:, -context_size:]
            
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.model(input_ids)[:, -1, :]                # logits for the last token
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)  # greedy pick
                input_ids = torch.cat([input_ids, next_token], dim=1)   # append
        output_text = self.tokenizer.decode(input_ids[0].tolist())
        
        self.model.train()
        return output_text
    
    
    def save_model_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
        
        
        
class loading_gpt_weights:
    """Class to load pretrainned gpt2 weights with an 
       aim to finetune the model later
    """

    def __init__(self, model, *args, **kwargs):
        super(loading_gpt_weights, self).__init__(*args, **kwargs)
        self.settings, self.params = download_and_load_gpt2(model_size="124M", 
                                                models_dir="gpt2")
        
        self.model = model
        self.model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
            }
        
    def assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                            f"Right: {right.shape}"
            )
        return torch.nn.Parameter(torch.tensor(right))
    
    def load_weights_into_astro_gpt(self):
        self.model.embed_layer.position_embedding.weight = self.assign(self.model.embed_layer.position_embedding.weight, self.params['wpe'])
        self.model.embed_layer.token_embedding.weight = self.assign(self.model.embed_layer.token_embedding.weight, self.params['wte'])
        
        for b in range(len(self.params["blocks"])):
            q_w, k_w, v_w = np.split((self.params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.model.transformer_blocks[b].attention.W_query.weight = self.assign(self.model.transformer_blocks[b].attention.W_query.weight, q_w.T)
            self.model.transformer_blocks[b].attention.W_key.weight = self.assign(self.model.transformer_blocks[b].attention.W_key.weight, k_w.T)
            self.model.transformer_blocks[b].attention.W_value.weight = self.assign(self.model.transformer_blocks[b].attention.W_value.weight, v_w.T)
            
            q_b, k_b, v_b = np.split((self.params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.model.transformer_blocks[b].attention.W_query.bias = self.assign(self.model.transformer_blocks[b].attention.W_query.bias, q_b)
            self.model.transformer_blocks[b].attention.W_key.bias = self.assign(self.model.transformer_blocks[b].attention.W_key.bias, k_b)
            self.model.transformer_blocks[b].attention.W_value.bias = self.assign(self.model.transformer_blocks[b].attention.W_value.bias, v_b)
            self.model.transformer_blocks[b].attention.out_proj.weight = self.assign(self.model.transformer_blocks[b].attention.out_proj.weight,self.params["blocks"][b]["attn"]["c_proj"]["w"].T)

            self.model.transformer_blocks[b].attention.out_proj.bias = self.assign(self.model.transformer_blocks[b].attention.out_proj.bias,self.params["blocks"][b]["attn"]["c_proj"]["b"])
            
            
            mlp = self.model.transformer_blocks[b].ffn.neural_net
            mlp[0].weight = self.assign(mlp[0].weight, self.params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            mlp[0].bias = self.assign(mlp[0].bias, self.params["blocks"][b]["mlp"]["c_fc"]["b"])
            mlp[2].weight = self.assign(mlp[2].weight, self.params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            mlp[2].bias = self.assign(mlp[2].bias, self.params["blocks"][b]["mlp"]["c_proj"]["b"])

            layer_norm_1 = self.model.transformer_blocks[b].layernorm1
            layer_norm_1.gamma = self.assign(layer_norm_1.gamma,self.params["blocks"][b]["ln_1"]["g"])
            layer_norm_1.beta = self.assign(layer_norm_1.beta,self.params["blocks"][b]["ln_1"]["b"])
            layer_norm_2 = self.model.transformer_blocks[b].layernorm2
            layer_norm_2.gamma = self.assign(layer_norm_2.gamma,self.params["blocks"][b]["ln_2"]["g"])
            layer_norm_2.beta = self.assign(layer_norm_2.beta,self.params["blocks"][b]["ln_2"]["b"])
            
        self.model.final_norm.gamma = self.assign(self.model.final_norm.gamma, self.params["g"])
        self.model.final_norm.beta = self.assign(self.model.final_norm.beta, self.params["b"])
        self.model.out_head.weight = self.assign(self.model.out_head.weight, self.params["wte"])
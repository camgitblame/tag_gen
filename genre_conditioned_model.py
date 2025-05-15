import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


class GenreConditionedModel(nn.Module):
    def __init__(self, base_model_name, genre_list, peft_config=None):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.genre_to_id = {genre: i for i, genre in enumerate(genre_list)}
        self.genre_embedding = nn.Embedding(len(genre_list), self.base_model.config.hidden_size)

        self.peft_model = get_peft_model(self.base_model, peft_config) if peft_config else self.base_model

    def forward(self, input_ids=None, attention_mask=None, labels=None, genre=None, **kwargs):
        # genre: list of genre strings for the current batch
        # Handle if genre elements are mistakenly lists (e.g., ['comedy'] instead of 'comedy')
        # genre is now a list of genre name lists: e.g., [['Comedy', 'Drama'], ['Action']]
        genre_ids = [
            [self.genre_to_id[g.lower()] for g in sorted(gs) if g.lower() in self.genre_to_id]
            for gs in genre
        ]

        # Pad genre_ids to same length (for batching)
        max_len = max(len(ids) for ids in genre_ids)
        padded = [ids + [0]*(max_len - len(ids)) for ids in genre_ids]
        mask = [[1]*len(ids) + [0]*(max_len - len(ids)) for ids in genre_ids]

        genre_ids_tensor = torch.tensor(padded, device=input_ids.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=input_ids.device)

        # Look up and average embeddings
        genre_embeds_all = self.genre_embedding(genre_ids_tensor)  # (batch, max_genres, hidden)
        masked = genre_embeds_all * mask_tensor.unsqueeze(-1)
        genre_embeds = masked.sum(dim=1) / mask_tensor.sum(dim=1, keepdim=True)  # (batch, hidden)
        genre_embeds = genre_embeds.unsqueeze(1)  # (batch, 1, hidden)

        inputs_embeds = self.peft_model.get_input_embeddings()(input_ids)  # (batch, seq, hidden)
        conditioned_inputs = torch.cat([genre_embeds, inputs_embeds], dim=1)

        # Extend attention mask to account for genre embedding
        extended_mask = torch.cat(
            [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask],
            dim=1,
        )

        # Fix mismatch between inputs and labels by padding labels with -100 at start
        if labels is not None:
            labels = torch.cat([
                torch.full((labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device),
                labels
            ], dim=1)

        outputs = self.peft_model(
            inputs_embeds=conditioned_inputs,
            attention_mask=extended_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    def get_input_embeddings(self):
        return self.peft_model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.peft_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.peft_model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        return self.peft_model.push_to_hub(*args, **kwargs)
    
    def generate_with_genre(self, input_ids, attention_mask=None, genre=None, **generate_kwargs):
        # === Same embedding injection as in your forward ===
        genre_ids = [
            [self.genre_to_id[g.lower()] for g in sorted(gs) if g.lower() in self.genre_to_id]
            for gs in genre
        ]
        max_len = max(len(ids) for ids in genre_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in genre_ids]
        mask = [[1]*len(ids) + [0]*(max_len - len(ids)) for ids in genre_ids]

        genre_ids_tensor = torch.tensor(padded, device=input_ids.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=input_ids.device)
        genre_embeds_all = self.genre_embedding(genre_ids_tensor)
        masked = genre_embeds_all * mask_tensor.unsqueeze(-1)
        genre_embeds = masked.sum(dim=1) / mask_tensor.sum(dim=1, keepdim=True)
        genre_embeds = genre_embeds.unsqueeze(1)

        # Get input embeddings and prepend genre
        inputs_embeds = self.peft_model.get_input_embeddings()(input_ids)
        conditioned_inputs = torch.cat([genre_embeds, inputs_embeds], dim=1)

        # Adjust attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = torch.cat(
            [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask],
            dim=1
        )

        # Now call generate with embeddings instead of input_ids
        return self.peft_model.generate(
            inputs_embeds=conditioned_inputs,
            attention_mask=extended_mask,
            **generate_kwargs
        )

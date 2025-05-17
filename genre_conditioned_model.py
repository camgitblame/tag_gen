import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


class GenreConditionedModel(nn.Module):
    def __init__(self, base_model_name, genre_list, peft_config=None):
        super().__init__()

        # Load the pretrained GPT-2 model with a causal language modeling head
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Map genre names to unique integer IDs
        self.genre_to_id = {genre: i for i, genre in enumerate(genre_list)}

        # Create an embedding layer to learn a dense vector for each genre
        self.genre_embedding = nn.Embedding(len(genre_list), self.base_model.config.hidden_size)

        # Apply LoRA adapters to the base model (if provided)
        self.peft_model = get_peft_model(self.base_model, peft_config) if peft_config else self.base_model
        self.genre_weight = 3.0  # 🔥 increase to boost genre impact (e.g., 1.0 = neutral, >1 = more influence)

    def forward(self, input_ids=None, attention_mask=None, labels=None, genre=None, **kwargs):
        # genre: list of genre strings per example, e.g. [['Comedy', 'Drama'], ['Action']]

        # Convert genre names to their corresponding IDs (sorted for consistency)
        genre_ids = [
            [self.genre_to_id[g.lower()] for g in sorted(gs) if g.lower() in self.genre_to_id]
            for gs in genre
        ]

        # Determine the longest genre list in the batch for padding
        max_len = max(len(ids) for ids in genre_ids)

        # Pad all genre ID lists to the same length with 0s
        padded = [ids + [0]*(max_len - len(ids)) for ids in genre_ids]

        # Create binary mask to track which positions are real genres (1) vs padding (0)
        mask = [[1]*len(ids) + [0]*(max_len - len(ids)) for ids in genre_ids]

        # Convert lists to tensors and move to the same device as inputs
        genre_ids_tensor = torch.tensor(padded, device=input_ids.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=input_ids.device)

        # Look up embeddings for each genre ID
        genre_embeds_all = self.genre_embedding(genre_ids_tensor)  # shape: (batch_size, max_genres, hidden_size)

        # Zero out the padding embeddings using the mask
        masked = genre_embeds_all * mask_tensor.unsqueeze(-1)

        # Average the genre embeddings per example (ignoring padding)
        genre_embeds = (masked.sum(dim=1) / mask_tensor.sum(dim=1, keepdim=True)) * self.genre_weight

        # Reshape genre embedding to add sequence dimension (for concatenation)
        genre_embeds = genre_embeds.unsqueeze(1)  # shape: (batch_size, 1, hidden_size)

        # Get token embeddings for input_ids from the model’s embedding layer
        inputs_embeds = self.peft_model.get_input_embeddings()(input_ids)  # shape: (batch_size, seq_len, hidden_size)

        # Prepend genre embedding to the token embeddings (acts as a special context token)
        conditioned_inputs = torch.cat([genre_embeds, inputs_embeds], dim=1)  # shape: (batch_size, seq_len+1, hidden_size)

        # Extend the attention mask to account for the extra genre token
        extended_mask = torch.cat(
            [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask],
            dim=1,
        )

        # Shift labels to match new input shape by prepending -100 (ignored in loss calculation)
        if labels is not None:
            labels = torch.cat([
                torch.full((labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device),
                labels
            ], dim=1)

        # Pass inputs into the model with the genre token and updated masks
        outputs = self.peft_model(
            inputs_embeds=conditioned_inputs,
            attention_mask=extended_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    # Utility methods to proxy common operations to the underlying PEFT-wrapped model
    def get_input_embeddings(self):
        return self.peft_model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.peft_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.peft_model.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        return self.peft_model.push_to_hub(*args, **kwargs)

    def generate_with_genre(self, input_ids, attention_mask=None, genre=None, **generate_kwargs):
        # Same steps as forward pass to prepare genre embeddings

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
        genre_embeds = (masked.sum(dim=1) / mask_tensor.sum(dim=1, keepdim=True)) * self.genre_weight
        genre_embeds = genre_embeds.unsqueeze(1)

        # Get token embeddings and prepend genre embedding
        inputs_embeds = self.peft_model.get_input_embeddings()(input_ids)
        conditioned_inputs = torch.cat([genre_embeds, inputs_embeds], dim=1)

        # Extend the attention mask to account for the added token
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = torch.cat(
            [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask],
            dim=1
        )

        # Generate tokens using the modified input
        return self.peft_model.generate(
            inputs_embeds=conditioned_inputs,
            attention_mask=extended_mask,
            **generate_kwargs
        )

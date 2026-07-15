# src/model.py

#import libraries

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
	""" Configuration du modèle small langage model (SLM) """

	vocab_size: int = 5000
	context_size: int = 128
	embedding_dim: int = 256
	num_heads: int = 4
	num_layers: int = 4
	dropout: float = 0.1

class FeedForward(nn.Module):
	"""Réseau feed-forward utilisé dans chaque bloc Transformer."""

	def __init__(self,config: ModelConfig):
		super().__init__()

		hidden_dim = 4 * config.embedding_dim

		self.net = nn.Sequential(
			nn.Linear(config.embedding_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, config.embedding_dim),
			nn.Dropout(config.dropout),
		)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

class CasualSelfAttention(nn.Module):
	"""Mécanisme d'attention causale utilisé dans chaque bloc Transformer."""

	def __init__(self, config: ModelConfig) -> None:
		super().__init__()

		if config.embedding_dim % config.num_heads != 0:
			raise ValueError("embedding_dim doit être divisible par num_heads")
		
		self.attention = nn.MultiheadAttention(
			embed_dim=config.embedding_dim,
			num_heads=config.num_heads,
			dropout=config.dropout,
			batch_first=True,
		)

		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		sequence_length = x.size(1)

		# True pour appliquer un masque causale
		casual_mask = torch.triu(
			torch.ones(
				sequence_length,
				sequence_length,
				device = x.device,
				dtype = torch.bool,
			),
			diagonal=1,
		)

		attention_output, _ = self.attention(
			query = x,
			key = x,
			value = x,
			attn_mask = casual_mask,
			need_weights = False,
		)

		return self.dropout(attention_output)
	
class TransformerBlock(nn.Module):
	"""Bloc Transformer combinant l'attention et le feed-forward."""

	def __init__(self, config: ModelConfig) -> None:
		super().__init__()

		self.norm_attention = nn.LayerNorm(config.embedding_dim)
		self.attention = CasualSelfAttention(config)
		
		self.norm_feedforward = nn.LayerNorm(config.embedding_dim)
		self.feed_forward = FeedForward(config)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Ajout de la connexion résiduelle pour l'attention
		x = x + self.attention(self.norm_attention(x))

		# Ajout de la connexion résiduelle pour le feed-forward
		x = x + self.feed_forward(self.norm_feedforward(x))

		return x

class SmallLanguageModel(nn.Module):
	"""Petit modèle de langage autorégressif de type GPT."""

	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		
		self.config = config

		self.token_embedding = nn.Embedding(
			config.vocab_size,
			config.embedding_dim,
		)

		self.position_embedding = nn.Embedding(
			config.context_size,
			config.embedding_dim,
		)

		self.position_embedding = nn.Embedding(
			config.context_size,
			config.embedding_dim,
		)

		self.dropout = nn.Dropout(config.dropout)

		self.blocks = nn.ModuleList(
			[TransformerBlock(config) 
			for _ in range(config.num_layers)
			]
		)

		self.final_form = nn.LayerNorm(config.embedding_dim)

		self.output_head = nn.Linear(
			config.embedding_dim,
			config.vocab_size,
			bias = False
		)
			
		# Initialisation des poids, partage des poids entre l'embedding et la tête de sortie

		self.apply(self._init_weights)

	def _init_weights(self, module: nn.Module) -> None:
		"""Initialise les poids du module."""

		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
		
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
	
	def forward(
			self,
			input_ids: torch.Tensor,
			targets: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		"""
		Calcule les logits et, si targets est foruni, la loss.

		inputs_ids :
			- Tensor de forme (batch_size, sequence_length) contenant les IDs des tokens d'entrée.

		targets :
			- Tensor de forme (batch_size, sequence_length) contenant les IDs des tokens cibles.
			- Si None, la loss ne sera pas calculée.
		"""

		batch_size, sequence_length = input_ids.size()

		if sequence_length > self.config.context_size:
			raise ValueError(
				f"La longueur de la séquence ({sequence_length}) dépasse la taille du contexte ({self.config.context_size})."
			)
		
		positions = torch.arange(
			sequence_length,
			device = input_ids.device,
		)

		token_embeddings = self.token_embedding(input_ids)
		position_embeddings = self.position_embedding(positions)

		x = token_embeddings + position_embeddings
		x = self.dropout(x)

		for block in self.blocks:
			x = block(x)

		x = self.final_form(x)

		logits = self.output_head(x)

		loss = None

		if targets is not None:
			loss = F.cross_entropy(
				logits.reshape(
					batch_size*sequence_length,
					self.config.vocab_size,
				),
				targets.reshape(batch_size*sequence_length),
			)

		return logits, loss
	
	@torch.no_grad()
	def generate(
			self,
			input_ids: torch.Tensor,
			max_new_tokens: int,
			temperature: float = 1.0,
			top_k: int | None = None,
	) -> torch.Tensor:
		"""
		Génère du texte de manière autoregressive.

		inputs_ids :
			- Tensor de forme (batch_size, sequence_length) contenant les IDs des tokens d'entrée.

		max_new_tokens :
			- Nombre maximum de tokens à générer.
		"""

		if temperature <= 0:
			raise ValueError("La température doit être strictement positive.")
		
		self.eval()
		
		for _ in range(max_new_tokens):

			# On tronque l'entrée si elle dépasse la taille du contexte
			context = input_ids[:, -self.config.context_size:]

			logits, _ = self(context)

			# On ne prend que le dernier token pour la génération
			next_token_logits = logits[:, -1, :]

			# Application de la température
			next_token_logits = next_token_logits / temperature

			# Application de top-k
			if top_k is not None:
				top_k = min(top_k, next_token_logits.size(-1))
				
				threshold = torch.topk(
					next_token_logits,
					top_k,
				).values[:,-1].unsqueeze(-1)

				next_token_logits = next_token_logits.masked_fill(
					next_token_logits < threshold,
					float("-inf"),
				)

			probabilites = F.softmax(next_token_logits, dim=-1)

			next_token = torch.multinomial(
				probabilites,
				num_samples=1
			)

			input_ids = torch.cat(
				(input_ids, next_token),
				dim=1,
			)

		return input_ids
	

def count_parameters(model: nn.Module) -> int:
	"""Compte le nombre de paramètres du modèle."""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model() -> None:
	"""Test rapide du modèle pour vérifier qu'il fonctionne correctement."""
	config = ModelConfig()
	model = SmallLanguageModel(config)

	batch_size = 2
	sequence_length = 32

	input_ids = torch.randint(
		low =0,
		high = config.vocab_size,
		size = (batch_size, sequence_length),
	)

	targets = torch.randint(
		low = 0,
		high = config.vocab_size,
		size = (batch_size, sequence_length),
	)

	logits, loss = model(input_ids, targets)

	print("Forme des entrées :", input_ids.shape)
	print("Forme des logits :", logits.shape)
	print("Loss de test :", loss.item() if loss is not None else None)
	print(f"Nombre de paramètres du modèle : {count_parameters(model)}")

if __name__ == "__main__":
	test_model()
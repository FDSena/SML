# src/tokenizer.py

#import libraries
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

INPUT_PATH = Path("data/input.txt")
OUTPUT_PATH = Path("data/tokenizer.json")

VOCAB_SIZE = 5_000

SPECIAL_TOKENS = [
	"<pad>",
	"<unk>",
	"<bos>",
	"<eos>",
]

def train_tokenizer() -> None:
	"""
	Entraîne un tokenizer BPE sur le texte d'entrée.
	"""
	if not INPUT_PATH.exists():
		raise FileNotFoundError(f"Le fichier d'entrée n'existe pas : {INPUT_PATH.resolve()}" "Lance d'abord le script prepare_data.py pour générer le fichier d'entrée.")
	tokenizer = Tokenizer(BPE(unk_token="<unk>"))
	tokenizer.pre_tokenizer = Whitespace()

	trainer = BpeTrainer(
		vocab_size = VOCAB_SIZE,
		special_tokens = SPECIAL_TOKENS,
		min_frequency = 2,
		show_progress = True,
	)

	tokenizer.train(files =[str(INPUT_PATH)], trainer=trainer)
	
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	tokenizer.save(str(OUTPUT_PATH))

	print(f"Tokenizer entraîné et sauvegardé dans : {OUTPUT_PATH.resolve()}")
	print(f"Taille du vocabulaire : {VOCAB_SIZE}")

	test_sentence = "Ceci est un exemple de phrase pour tester le tokenizer."
	encoded = tokenizer.encode(test_sentence)
	print(f"Phrase test : {test_sentence}")
	print(f"Tokens : {encoded.tokens}")
	print(f"IDs : {encoded.ids}")
	decoded = tokenizer.decode(encoded.ids)

	print(f"Phrase décodée : {decoded}")

if __name__ == "__main__":
    train_tokenizer()
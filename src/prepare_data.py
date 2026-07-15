# src/prepare_data.py

from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path("data/input.txt")
DATASET = "iproskurina/TinyStories-French"


def extract_text(example: dict) -> str | None:
    """Extrait le texte depuis un exemple du dataset."""
    for column in ("french-tinystories", "text", "story"):
        value = example.get(column)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def main() -> None:
    """Télécharge et prépare le dataset."""
    print(f"Chargement du dataset : {DATASET}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET, split="train")

    print(f"Colonnes disponibles : {dataset.column_names}")
    print(f"Nombre d'exemples : {len(dataset)}")

    stories_written = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in dataset:
            text = extract_text(example)

            if text is None:
                continue

            file.write(text)
            file.write("\n\n")
            stories_written += 1

    print(f"Nombre d'histoires écrites : {stories_written}")
    print(f"Fichier créé : {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

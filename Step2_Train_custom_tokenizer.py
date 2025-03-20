from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize a WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Add pre-tokenizer (splits text into words)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Initialize a trainer
trainer = trainers.WordPieceTrainer(
    vocab_size=20000,  # Smaller vocab size for 1.1 GB data
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train the tokenizer on your Malayalam data
tokenizer.train(files=["malayalam_text.txt"], trainer=trainer)

# Save the tokenizer
tokenizer.save("malayalam_tokenizer.json")

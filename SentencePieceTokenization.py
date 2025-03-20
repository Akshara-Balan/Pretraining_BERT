!pip install sentencepiece
import sentencepiece as spm

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.train(
    input="malayalam_text.txt",  # Path to your text file
    model_prefix="malayalam_sp",  # Prefix for the model files
    vocab_size=16000,  # Adjust based on your data size
    model_type="unigram",  # You can also use "bpe"
    max_sentence_length=128,  # Maximum sentence length
    pad_id=0,  # Padding token ID
    unk_id=1,  # Unknown token ID
    bos_id=2,  # Beginning of sentence token ID
    eos_id=3,  # End of sentence token ID
    pad_piece="[PAD]",  # Padding token
    unk_piece="[UNK]",  # Unknown token
    bos_piece="[CLS]",  # Beginning of sentence token
    eos_piece="[SEP]",  # End of sentence token
    user_defined_symbols=["[MASK]"],  # Add special tokens
)

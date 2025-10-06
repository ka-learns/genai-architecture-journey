# Week 1 Day 1

## Jupyter
- Allows you to run pieces of code without running an entire program, playground for learning how to work with LLMs
- Jupyter notebooks (.ipynb files) let you mix code, text, and output in a single place
- Run code cells one at a time instead of all at once
- Great for ML/AI

## Hugging Face
- open source hub for LLM, few lines of python to use LLM
- use `transformers` lib to download LLM with few lines of code
- from `transformers` lib import hugging face `pipeline` API to utilize pre-trained LLM
- syntax to load model: `generator = pipeline("pipeline", model="model")`
    - pipelines: text-generation, sentiment-analysis, summarization, translation_xx_to_xx (en to fr)
    - models
        - decoder only (text-generation): gpt2
        - encoder/decoder seq2seq: google/flan-t5-base
    - single library, different NLP, from pre-trained models
- syntax to run inference: `output = generator("Prompt", max_length=##, num_return_sequences=#)`
    - giving model a prompt
    - predict the next words, until `max length` in tokens
    - how many results you want with `num_return_sequences`
    - show models take input text
    - show how outputs depend on model size, randomness, and prompt
    - can also include `temperature` and `top_k` and `top_p`

# Week 1 Day 2

## Visualize Tokenization
- Understand how text becomes tokens that models process
- See what happens inside a tokenizer (wordpiece, subword, BPE)
- Grasp how tokenization affects context length, performance, and cost

### Types of Tokenization Algos
- Tokenization exists because LLM's can't process raw text, need numbers
- Tokenization is the process of mapping text to discrete integer tokens
- What should a token be?
    - a word? too many, each word becomes its own token
    - a character? too small with long sequences, inefficent
    - a subword? perfect, efficient and expressive
- Types of algos: BPE, WordPiece, and SentencePiece
- **Byte-Pair Encoding (BPE):**
    - Start from characters, repeatedly merge common pairs, building bigger chunks
    - Leads to subwords that balance efficiency and generalization
    - Rare words can still be broken into known smaller parts
    - Reduces unknown words (“unicornification” → “un”, “icorn”, “ification”)
    - Keeps common words as single tokens for efficiency
    - Compresses vocabulary
    - Used by GPT
- **Word-Piece:**
    - Let’s glue pieces that make the most sense in the whole sentence
    - Starts with characters, but instead of merging by most frequent pairs, it merges based on maximizing likelihood of the training under a model objective
    - Probabilistic version of BPE
    - Trade Offs: slightly better modeling of semantics, slightly slower to train
- **SentencePiece (Unigram LM):**
    - Treats the entire text as a raw stream (even spaces), learning tokens statistically via a unigram language model
    - Doesn’t rely on whitespace, works for languages without spaces
    - Trade-off: More flexible and multilingual, but tokenization may seem unintuitive to humans
- **Word-level tokenization:**
    - Splits only on spaces/punctuation
    - Rarely used now (too large vocab, too many OOV tokens)
- **Character-level tokenization:**
    - Each character = 1 token
    - Used in some special architectures (e.g., character CNNs, or multimodal models combining text + vision)
    - Super robust but inefficient for long text
- **Byte-level BPE:**
    - A variant of BPE (used in GPT-2 and GPT-3) that operates on bytes instead of Unicode characters
    - This allows it to handle any text (emojis, foreign scripts, control symbols) without special preprocessing
- Why It Matters to You (to an AI Architect)
    - Performance & cost: Models bill by tokens, knowing tokenization helps estimate usage
    - Context limits: Understanding tokens helps design prompts within model window (e.g., 4096 tokens)
    - Model interoperability: Different models = different tokenizers. Mixing them causes nonsense results
    - Debugging generation issues: When outputs truncate weirdly, it’s often due to tokenization mismatches


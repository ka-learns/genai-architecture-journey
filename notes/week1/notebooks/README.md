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

## Sentiment Analysis
- I downloaded a pretrained model BERT for sentiment analysis
- I then passed in a sample text to see how the LLM analyzed the sentiment, which came out amazing
- I then tried multiple inputs and it analyzed them correctly
- Tried another model ROBERTA
- It gave a good score, but the sentiment analysis labels were weird
- I wonder if I could modify the labels?
- It looks like the model is not pretrained on sentiment analysis, while it still can score it doesn't know how to label the score
- You can manually define or override the label mapping, example below
    - ``from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline``
    - ``model_name = "roberta-base"  # or your fine-tuned model``
    - ``model = AutoModelForSequenceClassification.from_pretrained(model_name)``
    - ``tokenizer = AutoTokenizer.from_pretrained(model_name)``
    - ``# manually define label mapping``
    - ``model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}``
    - ``model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}``
    - ``# create pipeline``
    - ``nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)``

# Week 1 Day 3

## Tokenization: Special Tokens, Padding, Truncation
- Understand how text is converted into numbers and structured for models
- Explore every token, padding, truncation, and special tokens
- ``bert-base-uncased`` is a model that knows English words in lowercase
- Explanation of Each Token:
    - Class Token (CLS): marks the start of a sentence, used for classification tasks
    - Separator Token (SEP): markes the end of a sentence, seperates sentences, used in question-answer or sentence-pair tasks
    - Padding Token (PAD): add extra tokens to make all sentences same length in a batch, empty spaces the model ignores
    - Markers that tell the model “start,” “end,” and “ignore”
- `input_ids`: numbers corresponding to words and special tokens
    - CLS = 101, SEP = 102, PAD = 0
- `attention_mask`: 1 for real tokens (pay attention), 0 for padding (ignore)
- `input_ids` and `attention_mask` are almost always present for batched inputs
- `token_type_ids` appear if the model architecture uses segment embeddings (BERT family), usually identify which segment this is from
    - have not been able to create any outputs with different segments
    - usually token type ids will be ignored or not even generated with other models, so I guess I am ok to skip this
- Padding: Makes all sequences the same length ``max_length=8`` by adding [PAD] tokens
- Truncation: Cuts sequences longer than ``max_length``
- In batching, all sentences should be same length
- Tensors are just multi-dimensional arrays, faster for math, GPU-friendly, and models expect them as input
    - pt = pyTorch: torch.Tensor objects (most common)
    - tf = TensorFlow: tf.Tensor objects (will be deprecated with Transformers lib)
    - np = numpy: numpy.ndarray arrays
    - none = python default

## GPU Validation
- Check if a machine has a GPU, learn how to borrow from Colab
- CPU is like a single helper who does many things slowly
- GPU is like a team of hundreds of helpers who can all work at once, large computations, parallelization
- GPUs make models learn thousands of times faster
- PyTorch: a library that helps computers do math with tensor objects very fast, especially using GPUs
- Test how long it takes torch.rand(10000,10000) and y = torch.matmul(x, x) to run
    - ``torch.rand(x,y)`` generate tensor object with random float numbers between 0.0 and 1.0 that is x columns and y rows long
    - used to generate dummy data in testing
    - 10000,10000 = generate 100 million numbers (10000x10000)
    - ``torch.matmul(x, x)`` 
    - multiply rows of one by columns of the other
    - 10000 x 10000 = 100,000,000 numbers * 10,000 calculations = 1 trillion operations 
    - Every time a neural network layer does its job, it’s basically doing this, but with huge matrices
        - Each input sentence → becomes a big list of numbers (a tensor).
        - Each layer of the model → has its own matrix of “weights” (numbers the model learned).
        - When you feed data in, the model multiplies the input matrix by the weight matrix to create new meaning — this is how it “thinks”.
        - combining information (x) with learned knowledge (y) to make predictions
- Helps to visualize how the performance of GPUs compares to CPUs when performing a trillion operations
- CPU took 24 seconds and GPU was less than 0.004 sec
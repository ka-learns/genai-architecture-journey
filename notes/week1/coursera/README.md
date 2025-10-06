# Module 1: Generative AI and LLM

# GenAI Basics
- **Generative AI:** focused on content creation, subset of traditional machine learning, analyzes statistical pattern in data created by humans to predict
    - Multiple modes: text, image, video, speech creation
- **Large Language Models (LLM):** models trained on trillions of words, with billions of parameters, using large amounts of compute power
    - Base Foundation Models: BERT, GPT, FLAN-T5 (open-source), PALM, BLOOM, LLaMA
    - Using LLMs, can remove the process of pre-training
- Interacting with LLM's is not ML programming
    - ML = computer code to interact with libraries
    - LLM = using natural language (prompts) to perform tasks
- Flow: human text -> prompt -> LLM -> output
- **Completion:** the output of the model, usually the output and text to respond
- **Inference:** act of using the model to generate the completion
- **Context Window:** the memory or space available to the prompt, usually few thousand words, but differs from model to model
- Use Cases: remember that GenAI & LLM are not just chat bots, there are many uses
    - Next Word Predictions, Translations (human/code)
    - Information Retrieval (entity recognition, word classification)
    - Augmenting LLM with data sources or APIs
- **Parameters:** memory allocated as a result of training, represents models subjective understanding of language
    - its ability to process, reason, and solve the task
    - The more the parameters increase, the more memory required, the more sophistication acheived

# LLM Architecture
- **Recursive Neural Network:** original architecture of LLM, found to be limited by compute and memory, not suitable for todays GenAI tasks
    - poor at prediction, even at scale, struggled with ambiguity
- **Convoluted Neural Network:** another older LLM architecture with limitations
- Ways to improve the architecture was studied in "Attention is all you need" paper, leading to todays LLM leaving RNN and CNN behind

## Transformer Architecture
- Developed as a result of the study published in "Attention Is All You Need"
- **Transformer Architecture:** newer architecture that speeds up the learning and understanding of words by LLMs by utilizing a self-attention mechanism, leading to todays LLMs
    - Look at words, understand relationships between them, and make predictions
- **Self-Attention:** the ability to see the connection between words by applying attention weights, improving the ability to encode the language
- **Attention Weights:** process of scoring relationships within a text, to understand the context and relevance of each word compared to the other words
- **Attention Map:** illustration of the attention weights between all of the words
- Transformer architecture helped to:
    - learn relevance and context of all the words in a sentence, in any order
    - ability to scale the model to multi-core GPUs: parallelize data to use bigger data sets, pay attention to the meaning of the words
- Simplified Transformer Architecture:
    - 2 distinct parts: encoder and decoder
    - Input -> Tokenization -> Embedding Layer -> Position Encoder -> Encoder Layer (Self-Attention Layer(s), Feed Forward Network) Iterations -> Decoder
    - Input -> Tokenization -> Embedding Layer -> Position Encoder -> Decoder (Encoder Output + Decoder Input, Feed Forward Network) -> Softmax Output -> Output
- ML models are like calculators, so words need to be converted to numbers to make the predictions
- **Tokenization:** the process of converting words to numbers then passed to the models
    - Each number is the numerical position of that word in a dictionary of all possible words
    - Multiple methods of tokenization: a token id for a complete word, or a token id for parts of a word
        - ex: Teacher: teacher = 100 (complete), teach = 100, er = 101 (part)
    - Important to keep the same tokenizer method from training in generation as well
- **Embedding Layer:** trainable vector embedding space, maps the tokenizer tokens into a vector with a unique location
    - Each token is mapped to a vector, each vector compared to other vectors, encodes the meaning/context of each token
    - The closer the vectors are to each other, the closer the relationship between the words
    - These vectors are then added to the bases of encoder/decoder as the position encoder
- **Position Encoder:** this vector stores the token/word order and relevance for parallized processing by the self-attention layer
    - The resulting vector is a combination of the token embedding vector + positional encoding vector
- **Encoder:** encodes input sequences into vector of deep representation of the structure and the meaning of the input
    - Consists of self-attention layer(s) and feed forward network, runs multiple iterations for all the tokens
    - Resulting vector is passed on to the the decoder
- **Self-Attention Layer(s):** mechanism that analyzes the relationships and contextual dependencies between the tokens in the input
    - Runs in both the encoder and decoder
    - Multiple iterations to understand context = multi-headed self-attention
- **Multi-Headed Self-Attention:** repeated iterations to understand context with self-attention layer, each iteration is considered an attention head
- **Attention Head:** each head is a different relationship, can be done in parallel and independently
    - Number of heads in the attention layer varies between model, 12-100 is the usual range
    - Examples of heads: entity, activity, rhyme, etc.
    - Cannot be defined, acquired through training
- **Attention Weight:** assigned to each relationship by attention head, scoring the significance of the head within the input sequence
- **Feed Forward Network:** the communication layer of the self-attention process
    - Stored in both the encoder (sends to output/decoder) and the decoder (sent to softmax output)
    - Processes the attention weights and outputs a vector of the probability of every token in the input sequence
- **Decoder:** uses the encoders contextual understanding to generate new tokens for prediction until the stop sequence is achieved
    - Consists of deep understanding and context from encoder, tokenized/positioned vector from the input
    - Multiple iterations from each input token to encoder vector, sent to fast forward network and aggregated in softmax output, until end of sequence
- **Softmax Output Layer:** normalizes received vector into proabability score for each token, output can have multiple probabilities with one being the highest
    - highest probability is the most likely predicted token 
    - scoring the output can be configured
- Final output is then detokenized into readable words

## Variations of the Transformer Architecture
- All use attention — the idea that some words are more important than others when understanding or generating text
- But how they use attention and masking (and what they’re trained to do) changes between the three architectures
- **Masking:** hiding some words or connections so the model can't cheat
    - **Token Masking:** hiding tokens randomly when training, to help the model understand context
    - **Attention Masking:** hiding future tokens, to prevent cheating when generating text
- **Encoder-Only Models (Autoencoding):**
    - Understanding-focused models that reads input to analyze and represent text
    - Bidirectional attention: every token is analyzed to all tokens (no attention masking)
    - Can predict the masked token after training (supports token masking)
    - Used for: classification, sentinment analysis, named entity recognition, Q&A
    - Not used for: Natural text prediction (not able to predict next words), translation or summarization (needs extra heads or fine-tuning)
    - BERT LLM
- **Decoder-Only Models (Autoregressive):**
    - Generation-focused models that read input and predict the token that comes next
    - Unidirectional attention: every token is analyzed to previous tokens (support attention masking - future words are hidden), look backwards
    - Predicts the next token without hiding the token (does not support token masking)
    - Used for: text generation (human/code), natural language flow, scales well with data
    - Not used for: understanding the full picture (can only predict the next word based on what it has seen), pure-text understanding (classification/sentiment)
    - GPT LLM
- **Encoder-Decoder Models (Sequence-to-Sequence):**
    - Input to Output translators, takes input sequence to produce another output sequence
    - Encoder reads full text (no masking) to get understanding, gives understanding to decoder, bidirectional attention
    - Decoder uses encoder understanding to write one token at a time (attention mask), unidirectional attention
    - Used for: translation, summarization, question answering, encoder helps decoder stay grounded
    - Not used for: more expensive model (computation), does not scale well compared to decoder only
    - FLAN-T5 LLM

# Attention Is All You Need White Paper
- Transformer Architecture revolutionized NLP and is the basis of LLMs today
- Replaced neural network architecture (RNN/CNN) with attention-based mechanism
- Self-attention can compute representations to capture long term dependencies within input and parallelize communication
- Encoder & Decoder layers, consisting of multi-head self-attention mechanism and feed forward network
    - Self-attention for segregating and handling input sequence
    - Feed forward network to connect layers to each positions seperately and identically
- Softmax Output layer to help residual connections and normalizing layers to help train and prevent overfitting
- Eliminates recurrent or convolutional operations with positional encoding to maintain order of the sequence

# LLM Pre-Training
- Modeling is code, where integration is prompts

## Prompt Engineering
- Flow: Prompt -> Inference -> Completion
- **Prompt:** input text, instructions for the task, the task itself, and expected output
- **Inference:** process of generating text based on understanding
- **Completion:** the output text, final step
- **Context Window:** full amount of text or memory available for prompt
- **In-Context Learning:** additional data within the prompt, provide examples in the context window for how you want the model to carry out the task
- **Zero-Shot Inference:** no influence on the exeuction of the task, prompt contains just the instructions, task, and expected output (big LLMs)
- **One-Shot Inference:** adding influence or guidelines for the task, you will include one example of the instructions and output, followed by the actual prompt and task
- **Two-Shot Inference:** adding more influence of guidelines for the task, you will include two examples of the instructions and output, followed by the actual prompt and task
- **Few-Shot Inference:** adding many exmaples of the instructions and output, followed by the actual prompt and task
- Shot inferences usually needed for helping smaller LLM's
- Adding shot inferences, will allow the model to learn how to execute the task as expected, learning by example
- Context window can limit the ability for in-context learning (scale, number of parameters, etc)
- If model is still failing after including 5-6 examples in few-shot inference, usually means more training is needed

## Training Configuration
- There are parameters that can be configured to help influence the way the model makes the final decision on the next token
- **Inference Parameters:** parameters that influence the output during inference
    - Can impact the number of tokens (max number of tokens), creativity of the output, during inference time invocation
    - Can be simple as max number of tokens, or more intricate such as limiting generated tokens (capping the number of iterations during selection process)
- **Training Parameters:** parameters learned during training time
- **Greedy Decoding:** used by most LLM by default, the process of choosing the next token based on highest probability
    - Works well with short generation, but susceptible to repeated words/sequences
- **Random Sampling:** non-default setting, choose output token based on pobability distribution to weight the selection
    - Helps generate more natural, non-repeating, creative tokens
    - But can also have a risk of wondering off or being incorrect
- Top K and Top P: sampling techniques to limit the random sampling, generates better output
    - **Top K:** randomly choose from only Top K tokens with the highest probability, allows randomness while limiting improbable tokens
    - **Top P:** limit random sampling to predictions whose cumulative probabilities do not exceed P, then choose randomly
- **Temperature:** scaling factor within the softmax output layer that influences the shape of the probability distribution that model calculates for next token
    - Higher temperature = higher randomness since broader and flatter probability distribution
    - Lower temperateure = less randomness since there is a strong-peaked distribution
    - Default temperature is 1

# GenAI Project Life Cycle
- A framework to take project from conception to launch
- Scope Project -> Select Model -> Adopt and Adapt Model -> Application Integration
- **Scope:** define the use case, make it accurate and narrow
    - Example: Single task, multiple task
- **Select:** decide whether to use and existing LLM or to train a new model from scratch
- **Adopt and Align Model:** assess the performance of the selection and train if needed
    - Training done with prompt engineering or in-context learning
    - Fine-Tuning (configurations, supervised learning)
    - Human Preferences (aligning with expected behavior)
    - Evaluate (metrics, benchmarks, performance)
    - Can include multiple iterations to get to desired state
- **Application Integration:** deploy into infrastructure and integrate with application
    - Optimization (compute resources, user experience)
    - Augementation (additional infrastructure required)

## Selecting the Best Model
- **Model Hubs:** centralized locations created by major framework developers to browse, explore model use cases, training methods, and known limitations
    - Hugging Face
- **Pre-Training:** self-supervised, initial training process for LLMs
    - the goal of this phase is to minimize the loss of the training objective, by updating model weights
    - deep statistical understanding of lanaguage developed in this phase
    - learns from unstructured data from many sources, requires lots of compute
    - internalizes patterns and structures within the language which enable model to complete task
- **Data Quality Filtering:** processing data to increase quality, address bias, and remove unwanted content
    - Consider filtering when estimating the amount of data, usually only 1-3% of tokens collected in pre-training
- Transformer models differ in training:
    - **Masked Langauage Modeling:** encoder only, training object to reconstruct original seqence after masking parts of input sequence, understand text deeply, bidirectional
    - **Denoising Objective:** encoder-decoder, training objective to fix or reconstruct corrupted input text, fix or transform text, bidirectional and unidirectional
        - **Span Corruption:** encoder side masks sequences of input tokens, decoder side reconstructs masked token sequences, outputs sentinel tokens with sequence tokens
        - **Sentinel Tokens:** tokens added to vocabulary that do not correspond with any actual word from input
    - **Causal Language Modeling:** decoder-only, training objective to predict next token based on previous sequence of tokens, generate new text, unidirectional
    - **Full Language Modeling:** hybrid replaced by CLM and MLM, predict every word using full context (not masked), learn language structure, bidirectional

# Training Challenges
- Common major training challenges are: Memory, Compute, Domain, and Time to Train
- Must factor all of these in when making the decision

## Memory Limitations
- Common error when training on NVIDIA GPUs: ``OutOfMemoryError: CUDA out of memory``
- **Compute Unified Device Architecture (CUDA):** collection of libraries and tools developed for NVIDIA GPUs
    - pyTorch, TensorFlow use CUDA to boost performance needed for deep learning operations
- Computers store numbers in 32-bit float (32-bit Full Precision), even this is not entirely accurate but close
- Memory Footprint for Training LLM w/ 32-bit Full Precision:
    - Model Parameters (Weights): 2 bytes per parameter
    - 2 Adam Optimizer States: 8 bytes per parameter (biggest footprint)
    - Gradients: 4 bytes per parameter
    - Activations/Temporary Memory (Variables): 8 bytes per parameter
    - Total (high-end): 24 bytes per parameter
    - Memory needed for storing 1 billion parameters = 4GB
    - Memory needed to train 1 billion parameters = 24GB (6X storage)
- As you can see with 32-bit full precision, memory becomes a huge limitation
- Numbers are stored as sign, exponent, fraction
- **Exponent:** range of numbers that can be stored
- **Mantissa/Significand:** fraction of the number, representing the precision
- **Quantization:** reducing memory to store parameters by reducing percision
    - Quantization from FP32 -> FP16 -> int8 reduces memory by half each step, but also reduces the precision
    - How can we reduce the memory, but maintain precision? BFLOAT16!
- **Brain Floating Point 16 (BFLOAT16):** alternative to FP16, optimization for 16-bit quantization, full dynamic range of FP32 with only 16 bits
    - reducing the memory but maintaining the precision
    - Developed by Google Brain, supported by new NVIDIA GPU (A100)
    - Popular choice for deep learning training (FLAN-T5 uses BFLOAT16)
    - Uses 1 bit for sign, 8 bits for exponent, 7 bits for fraction
    - Not well suited for rare integer calculations (not used in deep learning)
- Bits Needed to Store Numbers: (precision when storing pi)
    - 32-bit floating point (FP32): 1 bit for sign, 8 bits for exponent, 23 bits for fraction, 4 bytes (3.1415920257568359375)
    - 16-bit brain floating point (BFLOAT16): 1 bit for sign, 8 bits for exponent, 7 bits for fraction, 2 bytes
    - 16-bit floating point (FP16): 1 bit for sign, 5 bits for exponent, 10 bits for fraction, 2 bytes (3.140625)
    - 8-bit integer (int8): 1 bit for sign, 0 bits for exponent, 7 bits for fraction, 1 byte (3)
    - The range of numbers supported drops dramatically between options, int8 only supports -128 to +127
    - Range of numbers supported does not drop between FP32 and BFLOAT16
- **Quantization-Aware Training (QAT):** learn quantization scaling factors in training, supported by modern deep learning libraries
- Quantization can help with low billions of parameters, but can be an issue for LLMs nearing trillions of parameters
- As models get bigger memory is not the only issue, you will also need to distribute compute across GPUs

## Compute Limitations
- Example of single GPU training
    - Data Loader -> LLM on 1 GPU -> Update Model
    - As you can see it will have to iterate through all data synchrously, which is time consuming
- **Distributed Computing:** splitting compute tasks across GPU's for training
- **Multi-GPU Computing:** using multiple GPU's to speed up training
- **Data Parallelization:** processing different parts of the data set simultaneously to speed up training, to benefit from multi-GPU computing
- As you pass in batches of data, there is a chance that each GPU learns differently, Forward/Backward Pass and Gradient Synchronization help to keep the model consistent
    - **Forward Pass:** Each GPU will run the model against the data to generate an output, then calculate loss based on what it has learned, how wrong was it making predictions
    - **Backward Pass:** With the losses, compute local gradient via backpropogation, learn from mistakes to make better predictions
    - **Gradient:** computation to understand how much each weight contributed to loss, how can I make the best prediction given what I have learned
    - **Graident Synchronization:** the process where all GPU's share and average local gradients, share learnings to each GPYU
    - **Weight Update:** Each GPU updates its local copy of the model, update weights to make best prediction with new data set

### Distributed Data Parallel (DDP)
- **Distributed Data Parallel (DDP):** training method, copies entire model to each GPU, send batches of data to GPUs in parallel, followed by synchronization
- Granted that your model weights, gradients, and optimizer states are able to fit on a single GPU
- Popular training method, used by pyTorch
- Example: 4 GPUs, you can process the same data set in 1/4 of training time
- DDP Flow (4 GPU): 1/4 data set -> LLM Copy on GPU 1 -> Forward/Backward Pass -> Gradient Synchronization -> Weight Update (Update Model)
- Advantages: scales training time linearly with GPUs, consistent models across GPUs, easy to implement with pyTorch and TensorFlow
- Challenges:
    - Communication overhead with gradient synchronization, need faster interconnects (consider NVLink or inifiband)
    - Possible imbalance in workload (bad batching, slow GPU processing)
    - Model can be too big for GPU, redundant memory consumption with multiple copies of model (consider model sharding)

### Fully Sharded Data Parallel (FSDP)
- **Fully Sharded Data Parallel (FSDP):** popular sharding and data parallelization training method, where data and model are both sharded across GPUs
    - Speeds up training, using ZeRO redundancy
    - Reduces memory by sharding the model instead of replicating like in DDP
    - Data must be aggregated before forward/backward pass, can be maintained for future operations (performance vs. memory trade-off)
    - Supports offloading data to CPU if needed, reducing memory even more
    - Sharding factor is configurable to balance performance and memory
- **Model Sharding** not only split the data, but also shard the model (parameters, gradients, optimizer states) across GPUs, with no data redundancy
- **Zero Redundancy Optimizer (ZeRO):** optimize memory by distributing or sharding model across GPU's with zero data overlap
    - DDP has redundant memory since replicating model
    - ZeRO shards the model instead of replicating, focusing on the optimizer as it has the biggest memory footprint
    - 3 stages: ZeRO Stage 1, ZeRO Stage 2, and ZeRO Stage 3
        - **ZeRO Stage 1:** Pos, shards of optimizer states across GPUs, reduces memory by factor of 4
        - **ZeRO Stage 2:** Pos+g, shards of optimizer states and gradients across GPU's, combined with stage 1 reduces memory by factor of 8
        - **ZeRO Stage 3:** Pos+g+p, shards of optimizer states, gradients, and parameters across GPU's, combined with stage 1 and 2 for linear reduction in memory based on GPU count
- **Sharding Factor:** ability to control amount of sharding, ranging from 1 to total number of GPU's
    - 1 = no sharding = DDP = replication
    - Max = full sharding = most memory savings
    - Hybrid sharding: any number in between
- FSDP Flow (4 GPU): 1/4 data set -> 1/4 Model -> All Gather -> Forward Pass -> Reshard -> All Gather -> Backward Pass -> Reduce Scatter -> Update Model
    - In FSDP, each GPU only has a shard of the total model weights, optimizers, and gradients
    - To do forward/backward pass, each GPU needs the full model, so an all-gather is needed
    - **All Gather (Before Forward Pass):** each GPU aggregates model weights from other GPUs before forward computation
    - **Resharding (After Forward Pass):** process of releasing the aggregated model weights to respective GPUs after forward pass output (activation)
    - During all gather, each GPU will hold the total weights of the model
    - **All Gather (Before Backward Pass):** each GPU aggregates model weights from other GPU's before gradient computation
    - **Reduce-Scatter (After Backward Pass):** process of releasing the gradients and respective model weight shards back to respective GPUs, after gradient applied to own weights
- FSDP is a “gather → compute → scatter” dance repeated each pass
- Saves memory by only materializing full weights just in time for computation, and only keeping shards otherwise
- DDP and FSDP perform similary on smaller models, for bigger models DDP will OOM, where FSDP doesn't

### Compute-Optimal Models 
- Goal of pre-training is to maximize performance of learning objective while minimizing loss when predicting tokens
- To increase training you can increase the data set and increase the model parameters, there are compute and time restraints that need to be considered
- How do you measure compute? Floating Point Operations per second (FLOP/s)
- How do you measure time with compute? Floating Point Operations per second per day (FLOP/s per day)
- **PetaFLOP:** quadrillion floating point operations
    - 8 NVIDIA V100 GPUs or 2 NVIDIA A100 GPUs can support this rate for a day
- Cost to support high petaflops is high, showing how compute budget plays a role
- With more parameters, it takes more days to train, showing how time is a constraint
- 3 Training Scaling Choices: dataset size, model size, compute budget
- **Power Law Relationship:** mathematical relationship between 2 variables, where one variable is proporational to another raised to some power, logarithmic linear line
    - dataset size, model size, compute budget - all have power law relationships with another variable, with the third held constant
    - example: all show test loss down and performance up
- What is the ideal balance then? Chinchilla Paper of 2022
- **Chinchilla Paper of 2022:**
    - solved the question of optimal data set compared to model size on a fixed compute budget
    - Most LLM today are over-parameterized and under-trained
    - Optimal Model Parameter Size:Data Set Size = 20:1 (70B Parameters: 1.4T Token Size)
    - Smaller models with less parameters have better results than bigger models with more parameters
    - Also showed that pre-training your own model is more beneficial

## Real-World/Domain Limitations
- Uncommon language structures like medical, legal, finance worlds
- Pre-Training your own model is faster and more performant: BloombergGPT
- Domain can also create additional trade-offs
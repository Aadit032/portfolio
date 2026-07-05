"How hard can this be?" 
To satisfy my curiosity and understand every design choice firsthand, I built a 25M parameter GPT from scratch.
This article is a collection of the lessons I learned along the way.
TLDR
Trained a 25M parameter LLM from scratch on 500M tokens (FineWeb + TinyStories) using modern techniques such as RoPE, GQA, SwiGLU and RMSNorm on NVIDIA RTX A5000 GPU for ~2 hours.
Model weights along with training results and logs can be found on HuggingFace: https://huggingface.co/Aadit-032/LiteGPT-25M
Full code for the training can be found on Github: https://github.com/Aadit032/Lite-GPT
Why bother building this?
Building this from scratch taught me a lot about the importance of data during the pre-training phase, it gave me insight into scaling laws and helped me demystify what had previously felt like a black box.
I got the inspiration from @karpathy NanoGPT which made me think about how I could improve upon it by using a different architecture or creating a different mix of datasets. I left the video feeling excited about the experiments I could run and wanted to push the limits by getting better results with fewer parameters and compute.
The repo is built to be easy to read for beginners while also being flexible enough for future experiments.
Section 1: Architecture
The previous version of this model LiteGPT-16M followed the GPT-2 style and had LayerNorm, MHA,  GELU FFN and learned positional encoding.
For the 25M model, I decided to modernize the architecture and use things like RMSNorm, SwiGLU FFN, RoPE and GQA. These methods make training a lot faster and scale better.

1.1) Dataset
I started with a dataset mixture containing 60% Fineweb, 30% TinyStories and 10% The Stack Smol from HuggingFace because I wanted to fine-tune the base model later to create a small coding agent for fun but I had to remove the code dataset completely as I wanted the model to understand natural language better and the coding data introduced a very different token distribution than natural language and I thought that it would help the model generalize better.
The dataset I finally settled on and where the model was finally trained on was 60% FineWeb and 40% TinyStories. I expected the loss to go down by a lot after I removed the The Stack Smol dataset keeping only natural language data but this didn’t change the loss by too much.
Next experiment would be to train this model exclusively on FineWeb or TinyStories. I am quite sure I can get the model’s validation loss to go down.
1.1.1) Tokenizer
I started out by using the GPT-2 tokenizer(same as NanoGPT) and I couldn’t get the validation loss below 4 which suggested something was fundamentally wrong and I realized that its n_vocab = 50257, which means my embedding layer is n_vocab x d_model and at that time my d_model = 320 (because I was still trying to train on the free-tier T4 GPU).
ie. 50257 x 320 ~ 16M out of the 25M trainable params that I had.
To fix this, I trained my own byte-level BPE for ~16K vocab_size using Tokenizers library from HuggingFace.
The dataset is tokenized and stored in uint16 format in .bin files and is loaded lazily when required with the help of np.memmap() that creates a memory mapped view allowing the data loader to grab random chunks without loading the whole file into memory.
500M tokens at uint16 is 500,000,000 x 2 bytes ~ 1GB
np.memmap() allows loading files bigger than the RAM and has a low startup time because it only loads the pages that are being read, directly from the disk.
1.2) Scaling laws
Before Chinchilla, models like GPT-3 were undertrained (175B params trained on ~300B tokens).
The chinchilla scaling laws suggest,  C∝N×D, where C = compute, N = Number of parameters and D = Number of training tokens. This means for a fixed amount of compute, to increase one you have to decrease the other.
Chinchilla suggests that the compute-optimal regime is approximately 20 training tokens per parameter.
1.3) Learning rate and cosine decay
The learning rate is not a constant value and is linearly warmed up to the max learning rate of 6e-4 for the last training run. This is done because when the model is initialized with random weights the initial losses are very large and if the learning rate is kept at a high value from the start, it will cause the updates to explode.
To mitigate that, we warm up the learning rate to its peak value for roughly 10% of the max steps and then decay the learning rate, so that by the end, when the loss of the model is converging, the updates will get smaller and smaller, helping the loss to converge to a smaller value more smoothly.
1.4) Transformer
1.4.1) RMSNorm
I have used the GPT-2 style pre-norm architecture where it first applies RMSNorm then goes to the attention block.
R
M
S
N
o
r
m
(
x
)
=
x
1
d
∑
i
=
1
d
x
i
2
+
ϵ
⊙
γ
RMSNorm normalizes only by the root-mean-square magnitude and does not subtract the mean, making it computationally cheaper than LayerNorm.
1.4.2) RoPE
Before attention, the K and Q vectors are rotated by m\theta_i where, theta_i = 10000^{-2i/d} to encode positional information before attention, this rotation allows the attention mechanism to model relative distances between the words.
R
(
θ
)
=
[
cos
⁡
θ
−
sin
⁡
θ
sin
⁡
θ
cos
⁡
θ
]

q
′
2
i
=
q
2
i
cos
⁡
(
m
θ
i
)
−
q
2
i
+
1
sin
⁡
(
m
θ
i
)
q
′
2
i
+
1
=
q
2
i
sin
⁡
(
m
θ
i
)
+
q
2
i
+
1
cos
⁡
(
m
θ
i
)
Earlier Transformers used fixed sinusoidal positional encodings, while GPT-2 switched to learned positional embeddings. More recent models such as LLaMA use RoPE. I decided to use RoPE because it allows attention to model the relative positions between tokens. It also reduces the need for more learned parameters and it generalizes better to contexts longer than those seen during training.
1.4.3) GQA Flash Attention
The attention block implements Grouped Query Attention (GQA), where there are n_heads query heads but only n_kv_heads key and value heads. Multiple query heads share the same key and value heads, significantly reducing the memory and computation required for the KV cache during inference.
Different attention heads learn different attention patterns. Empirical studies have shown that sharing keys and values across multiple query heads has little impact on model quality, making GQA an effective trade-off between efficiency and performance.
Attention is computed using torch.nn.functional.scaled_dot_product_attention(), which dispatches to highly optimized fused kernels that combine the attention operations into a single implementation, reducing memory accesses and improving throughput. When the hardware and input satisfy the required conditions, PyTorch automatically uses FlashAttention, providing further speedups and lower memory usage without requiring changes to the model code.
A
t
t
e
n
t
i
o
n
(
Q
,
K
,
V
)
=
S
o
f
t
m
a
x
(
Q
K
⊤
d
k
)
V
1.4.4) Residual add
After the attention block, the original input is added back to the output of the attention layer through a residual connection.
y
=
x
+
A
t
t
e
n
t
i
o
n
(
R
M
S
N
o
r
m
(
x
)
)
Residual connections preserve the original representation while providing an path for gradients, making optimization significantly easier. They provide a direct path for gradients to flow during backpropagation, helping avoid the vanishing gradient problem as models become deeper. This allows each transformer block to learn a small refinement to the input instead of having to learn an entirely new representation from scratch, resulting in faster convergence and more stable training.
The same residual connection is also applied after the SwiGLU feed-forward network, making every transformer block responsible for learning only incremental improvements to the representation.
1.4.5) SWiGLU MLP / FFN
Then comes the SwiGLU MLP, here, instead of one up-projection as done in GELU, there are 2 up-projections, one is the original up-projection and the other is the gate projection. This gating feature decides which features are worth keeping and which aren’t and suppresses those features thus acting as a feature gate.
S
w
i
G
L
U
(
x
)
=
S
w
i
s
h
(
x
W
+
b
)
⊗
(
x
V
+
c
)
1.4.6) Cross entropy loss
After going through n_layers of transformer blocks, it goes through a final RMSNorm and produces the final logits which are then passed through softmax to return probabilities over n_vocab choices.
The loss is calculated here using cross entropy loss which takes the negative log probability assigned to the correct next token.
L
=
−
1
N
∑
i
=
1
N
log
⁡
(
p
y
i
)
where p_y is the probability assigned to the correct token.
The negative log is taken because it strongly penalizes when the model is confidently wrong. The farther the probability is from the correct answer, the stronger the penalty is.
1.4.7) Generation
The model doesn’t output probabilities by default, when the model is being used for inference, these logits are divided by the temperature and passed through a softmax function that gives us the probabilities of the next token over the entire vocabulary of the model.
S
o
f
t
m
a
x
(
z
i
)
=
e
z
i
/
T
∑
j
e
z
j
/
T
Temperature controls the sharpness of the output probability distribution. Higher temperature produces a flatter distribution with more similar probabilities, while lower temperature produces a sharper distribution where the highest-probability tokens dominate.
Section 2: Implementation
This is the model configuration:
The d_hidden is kept approximately 8/3 x d_model following the LLaMA-style models instead of the 4 x d_model convention of the GPT-2 models. This is done because I am using SwiGLU which uses 3 linear projections instead of the 2 projections of GELU used in GPT-2.
Parameter count:
Training ran for 40,000 forward/backward iterations with a gradient accumulation factor of 2, resulting in 20,000 optimizer steps.
Total tokens seen:
64 × 512 × 40000 = 1,310,720,000 ≈ 1.3B tokens
That's roughly 2.6 epochs over the 500M-token dataset.
Section 3: Issues faced
3.1) Hardware Issues
I was using the free tier T4 GPU on Google Colab but it was too slow and google terminates the sessions quite aggressively so I was unable to finish my training completely. It also forced me to use a smaller batch size as a 16GB VRAM could not fit a bigger batch.
The memory used by a model in training is roughly:
Weights + Gradients + Optimizer states + Activation
I then switched to using Runpod and rented a NVIDIA RTX A5000 GPU, allowing me to increase the batch size, hidden dimension, and several other hyperparameters.
There are 2 ways to reduce the VRAM used on GPUs:
Gradient checkpointing: Instead of storing all the activations of the model, store only a few activations and when backpropagating, recompute the activations.
Gradient accumulation: This increases the effective batch size by accumulating gradients over multiple forward/backward passes before performing one optimizer update.
3.2) Train loss not going down
The loss was not budging and the model was not able to get to a good loss value, which meant that something in my setup was broken.
After spending quite some time looking around I realized that my tokenizer had n_vocab of 50257 and the embedding layer ended up taking 16M params out of the 25M trainable ones and that’s why the model was just not able to generalize all that data when the transformer only had 9M params to train on.
3.3) I thought number of steps is a good way to know if the model has trained enough
This one is embarrassing but I thought that training the model for more no. of steps will improve it… without changing the parameter count or increasing the dataset size. I learned the hard way that the number of quality tokens seen throughout training is a much better metric to focus on.
I was running my model through a small dataset for a lot of epochs which can be seen from the model card for LiteGPT-16M. The train loss was ~ 2 and the val loss was ~ 5 which meant that the model overfitted on the training dataset and failed to generalize the information contained within it.
3.4) Wrong data mix
The initial data mix containing coding data was not appropriate for this model size. It failed at generating anything that resembled code and the model couldn’t generalize on it.
3.5) Training on a small dataset
Before I knew anything about these scaling laws, I was training this model on very little data which led it to overfitting on the train data and it memorized the training data instead of learning representations that generalized to unseen data.
Section 4: Improvements to make
Train exclusively on FineWeb (larger, more diverse dataset)
Scale to more tokens (>1B) following Chinchilla scaling laws
Fine-tune on instruction data for a specific downstream task
Try alternative optimizers (AdamW-mini, Sophia)
Section 5: Results
Final Metrics
GPU Benchmark (Inference on A5000)
Section 6: Example generations
Prompt: The capital of France is
well known for its great geographical experience. The historical setting is not one of a sort of the greatest of all the times, but it's a central part of the history of the country. It's a historical town that has seen a huge shift inside this famous city of the same name. The city is known as "The Dolceus" and also known as "The Dolceus." It's a place where a lot of people live and live.
Prompt: Once upon a time, there lived
a little girl named Lily. She was very excited because she was going on a trip to see Grandma. She helped her mom pack their luggage with clothes and toys. Then, they got in the car and drove to Grandma's house. As they drove, Lily saw something strange. It was a big, gray cloud that seemed to float away. She pointed and asked her mom what it was. Her mom said it was a mild cloud, but it was a very hot day.
Prompt: The dragon opened its eyes and
looked around. It saw something shiny and colorful. It jumped and grabbed the object with its sharp claws. The dragon flew away and the dragon breathed a sigh of joy. It flew back up into the sky and found its owner, who was so proud of it. He said the dragon had been an amazing hunter.
Prompt: Tom had a little red ball.
Lily had a red ball and Tom had a blue ball. They both loved to play with their toys. But they both wanted the red ball. They both pulled it from each other. "Give me the red ball!" Lily shouted. "It's mine!" "No, it's mine!" Tom yelled. "It's mine!" They pulled so hard that they separate the red ball from each other. They both fell down and started to cry. Mom heard the noise and came to see what was wrong. She saw the red ball on the floor and the tears on Lily's face. She was not angry, but she was sad. "Why are you fighting?" Mom asked. "You both have the red ball, but you both have the same toy." "That is boring!" Lily said. "You both need to share and take turns." "Okay!" Mom said. "But you can both have the red ball.
Section 7: Observations
The model performs best on story-like prompts (not surprising given the 40% TinyStories training data)
Factual accuracy is poor at 25M parameters — it lacks the capacity to store facts
Repetition loops are common at higher temperatures
Coding generations are essentially non-functional (too few parameters for structured output)
Reasoning is limited but shows some grasp of narrative structure
Section 8: References
Attention Is All You Need (Vaswani et al., 2017)
NanoGPT (Karpathy)
Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019)
LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)
Scaling Laws for Neural Language Models (Kaplan et al., 2020)
Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022)
FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
RMSNorm: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)

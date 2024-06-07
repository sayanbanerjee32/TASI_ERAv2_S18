# TASI_ERAv2_S18

## Objective

Train your transformer (encoder-decoder) in a way such that
1. Loss should be less that 1.8 in 18 epochs
2. Accelerate your transformer training significantly (can use smart batching, One Cycle Policy, Automatic Mixed Precision etc),  

## Dataset - opus_books

- The dataset is sources from https://huggingface.co/datasets/Helsinki-NLP/opus_books
- This is a collection of copyright free books aligned by Andras Farkas, which are available from http://www.farkastranslations.com/bilingual_books.php
- The dataset supports 16 different languages.
- The dataset is setup as translation between 2 languages. e.g. for English to Italian `{ "en": "\"Jane, I don't like cavillers or questioners; besides, there is something truly forbidding in a child taking up her elders in that manner.", "it": "— Jane, non mi piace di essere interrogata. Sta male, del resto, che una bimba tratti così i suoi superiori." }`
- This experiment uses only the English - Italian pair of sentences, which contains around 32k rows.


## Techniques Used for Acceleration
### Smart Batching
#### Dynamic Padding: [code](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/blob/main/dataset.py)
Here sequence length (that would be input to the encoder) of each batch is decided based on the maximum length of sentence present in the batch. The sentences shorter than the longest sentence in that batch are only padded to match up to that length. 
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/assets/11560595/0c350f4d-2e90-417c-b6ef-0579c3cca023)

#### Uniform Length Batching: [code](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/blob/main/smart_batching_dataset.py)
Here sentences are first sorted based on length and then batches are created keeping similar length sentences are together. After this step the dynamic padding is applied.
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/assets/11560595/6746e398-7c0a-494b-9710-9a464499a1a6)


### One Cycle Policy
One Cycle policy is applied at the time of training. The Learning rate is increased till 0.0001 in first 30% of training iteration and then annealed in the rest. Following graph shows how the learning rate is updated throughout the iterations.
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/assets/11560595/f807be64-c89d-484f-a5d7-0226c0cb1e39)


### Optimiser
Experimented with both Adam and Liao optimiser and observed that Lion optimiser has helped in converging faster.

### Automatic Mixed Precision
`torch.cuda.amp.autocast` is used to run training mixed precision. While the forward pass is done in fp16, backpropagation still happened in fp32. To prevent underflowing gradients (i.e., gradients are too small to take into account) due to use of fp16, scaled gradients by some factor (so they aren't flushed to zero) using GradScaler. 


## Results
### Dynamic Padding
Notebook available [here](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/blob/main/S18_SayanBanerjee_dynamic_batching.ipynb)
- Time taken for each batch: 4 minutes
- Loss after 18 epochs: 1.687

### Uniform Length Batching
Notebook available [here](https://github.com/sayanbanerjee32/TASI_ERAv2_S18/blob/main/S18_SayanBanerjee_Uniform_batching.ipynb)
- Time taken for each batch: 2 minutes 15 seconds
- Loss after 18 epochs: 2.8 (Could not meet the criterion)

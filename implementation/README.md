# Implementation

We aim to implement the algorithm specified in this [paper](./research-paper.pdf). The main code for the implementation resides in [main.py](./main.py) and [main.ipynb](./main.ipynb). Some short notes on the implementation can also be found in [notes.pdf](./notes.pdf)!

## Basic Overview of the Algorithm

1. Load the tokenizer and the model
1. Construct the prompt (instructions + variable part) and initialize the variable part with random tokens
1. Tokenize the prompt and convert it to one-hot encoding
1. Initialize the optimizer and scheduler
1. Start the training
   - Calculate the outputs for the prompt (one-hot form)
   - Calculate the loss between this output and the expected output (labels)
   - Calculate the gradients
   - Set the gradients for the instructions part of the prompt to 0 since that should not be changed by the algorithm
   - Take a step along the gradient
   - Project the new prompt onto the simplex (simplex + entropy projection)
   - Pick one output from the available ones using `top_p` sampling and `GumbelSoftmax` (only for the variable part of the prompt, in one-hot form)
   - Update the prompt with this new variable part
1. Decode the updated prompt tokens to get the final prompt for breaking the LLM!

## TODO

- [x] Add basic implementation
- [ ] Shift to using one-hot encodings for generating logits
- [ ] Implement masking of inputs between runs

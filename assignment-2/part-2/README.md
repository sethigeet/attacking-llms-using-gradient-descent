# Assignment 2 - Part 2 - Testing Various Prompts on Various Models

Every language model has a signature prompt on which the model is fine-tuned on and hence gives the best results on it. This assignment aims to test out various prompts on various models and see how they perform when that prompt is no longer the signature one.

The code for this assignment is in [this notebook](./testing-various-prompts-for-various-models.ipynb) while the outputs for each of the prompts on each of the models is kept in [outputs.md](./OUTPUTS.md).

## Models Used

- `lmsys/vicuna-7b-v1.5`
- `openai-community/gpt2`
- `microsoft/wavecoder-ultra-6.7b`
- `Qwen/Qwen2-7B-Instruct`

## Prompts Used

The prompts and the question asked has been specified along with the outputs in [outputs.md](./OUTPUTS.md)!

## Conclusion

From the outputs, we can see that `openai-community/gpt2` being the smallest model among the ones being tested, shows its shortcomings and its outputs are some of the worst among the bunch while `Qwen/Qwen2-7B-Instruct` gave some of the best outputs (which proves why it is ranking at the top of the leader board among open source LLMs)

## GDGPT

### gdgpt

- Follows the gd model exactly (layernorm and dropout applied, but mathematically backed)

### gdgpt_plus

- Adds additional regularizations/normalizations that aren't mathematically explained in order to more closely follow the full gpt model

## GPT

### gpt

- An implementation of gpt that is missing some common regularization/normalization techniques to more closely follow the gd setup

### gpt_plus

- A complete (though simple) gpt model designed for performance

## Other

# ffm

- Exclusively embeddings and a ff to gauge the power of the ff

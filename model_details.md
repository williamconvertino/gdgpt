## GDGPT

### gdgpt_min

- Minimal implementation of gdgpt
- Perfectly follows mathematical model, no regularizations

### gdgpt

- Follows a slightly more complex mathematical model (layernorm and dropout applied, but mathematically backed)

### gdgpt_plus

- Adds additional regularizations that aren't mathematically explained in order to maximize performance

## GPT

### gpt_min

- Minimal implementation of gpt
- No regularizations
- Comparable to gdgpt_min

### gpt

- Added regularizations
- Comparable to gdgpt

### gpt_plus

- A complete gpt model
- Comparable to gdgpt_plus

## Other

# ffm_min

- Exclusively embeddings and feed forward network (no attention)
- Comparable to gdgpt_min

# ffm

- Same as ffm, but with regularizations
- Comparable to gdgpt

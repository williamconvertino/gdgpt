import torch
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import json

from src.util import get_time_remaining

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/evaluations/input')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/evaluations/output')

MODEL = 'gpt-4o-mini'
FILE_NAME = f"{MODEL}_eval_input.jsonl"
BATCH_ID = 'batch_678594cc4a008190b4ffc6ae686aecc8'

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.env')
assert os.path.exists(env_path), ".env file not found at {env_path}."

load_dotenv(env_path)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

assert OPENAI_API_KEY is not None, "OpenAI API key not found in .env file."

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = "You are an expert writing evaluator designed to assess student story completions. Your role is to provide constructive, fair, and detailed evaluations based on specific rubric criteria."

USER_PROMPT = """
In the following exercise, the student is given a pre-written beginning of a story. The student needs to complete this story. The exercise tests the student´s language abilities and creativity.

Here is the pre-written beginning:

<PROVIDED BEGINNING>
[STORY_BEGIN]
</PROVIDED BEGINNING>

And here is the students response:

<STUDENT RESPONSE>
[STORY_END]
</STUDENT RESPONSE>

First, provide a concise qualitative assessment about the student's writing. Then, give the writing a grade out of 10. These assessments should be done for each of the following rubric items:

1. Grammar:
* Is the writing grammatically correct?
* Evaluate syntax, punctuation, and sentence structure.
2. Consistency:
* Is the student's writing consistent with the provided beginning of the story?
* How well does the student complete the final sentence of the prescribed beginning?
3. Plot:
* Does the plot of the student's writing make sense (regardless of the provided beginning)?
4. Creativity: 
* How creative is the student's writing?

Format your response as follows:

<GRAMMAR>
[Qualitative assessment of grammar]
</GRAMMAR>
<GRAMMAR_GRADE>
[Grade out of 10]
</GRAMMAR_GRADE>

<CONSISTENCY>
[Qualitative assessment of consistency]
</CONSISTENCY>
<CONSISTENCY_GRADE>
[Grade out of 10]
</CONSISTENCY_GRADE>

<PLOT>
[Qualitative assessment of plot]
</PLOT>
<PLOT_GRADE>
[Grade out of 10]
</PLOT_GRADE>

<CREATIVITY>
[Qualitative assessment of creativity]
</CREATIVITY>
<CREATIVITY_GRADE>
[Grade out of 10]
</CREATIVITY_GRADE>

Provide your assessment below:
"""

def get_request_object(custom_id, content):
    return {
        "custom_id": f"{custom_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            "max_tokens": 1000
        }
    }


def generate_gpt4o_inputs(model, tokenizer, test_dataset, num_generations=10):
  
  i = 0
  num_skipped = 0
  start_time = time.time()
  
  eval_items = []
  
  for i, batch in enumerate(test_dataset):
    
    if i >= num_generations:
      break
    
    sequence = batch[0]
    if tokenizer.eos_token_id in sequence:
      eos_index = len(sequence) - sequence.tolist()[::-1].index(tokenizer.eos_token_id)
      sequence = sequence[eos_index:] # Trim sequence to include only the most recent story
    
    input_size = model.config.context_size // 2

    model_input = sequence[:input_size]
    
    with torch.no_grad():
      model.eval()
      
      story_begin = tokenizer.decode(model_input.tolist())
      story_true_end = tokenizer.decode(sequence[input_size:].tolist())
      
      beam_search_sequence = model.beam_search(model_input.unsqueeze(0), eos_token=tokenizer.eos_token_id)
      
      beam_search_sequence = beam_search_sequence[0, input_size:].tolist()
      if tokenizer.eos_token_id in beam_search_sequence:
        print(f"EOS token found in beam search sequence {i}.")
        exit()
        # print(f"Beam: {tokenizer.decode(beam_search_sequence)}")
      
      if len(beam_search_sequence) < 2: # Exclude sequences with less than 2 tokens, to avoid confusion in the GPT-4o evaluation
        print(f"Skipping sequence {i} due to insufficient length.")
        print(f"Prompt: {tokenizer.decode(model_input.tolist())}")
        print(f"Beam: {tokenizer.decode(beam_search_sequence)}")
        num_skipped += 1
        continue

      story_beam_end = tokenizer.decode(beam_search_sequence)
      
      true_prompt = USER_PROMPT.replace('[STORY_BEGIN]', story_begin).replace('[STORY_END]', story_true_end)
      beam_prompt = USER_PROMPT.replace('[STORY_BEGIN]', story_begin).replace('[STORY_END]', story_beam_end)
      
      eval_items.append(get_request_object(f"request_{i}_true", true_prompt))
      eval_items.append(get_request_object(f"request_{i}_beam", beam_prompt))
      
      i += 1
      time_remaining = get_time_remaining(start_time, i, num_generations)
      print(f"\r{i}/{num_generations} ({100 * i / num_generations:.2f}%) | Time Remaining: {time_remaining}", end='')
    
  os.makedirs(INPUT_DIR, exist_ok=True)
  with open(f'{INPUT_DIR}/{FILE_NAME}_input.jsonl', 'w') as f:
    for item in eval_items:
      f.write(f"{json.dumps(item)}\n")
  print(f"Generated inputs for GPT model:{MODEL}\n Processed {i}, skipped {num_skipped}.")
  
def create_batch():
  batch_input_file = client.files.create(
    file=open(f'{INPUT_DIR}/{FILE_NAME}_input.jsonl', 'rb'),
    purpose="batch"
  )
  batch_input_id = batch_input_file.id
  batch = client.batches.create(
    input_file_id=batch_input_id,
    endpoint="/v1/chat/completions",
    completion_window='24h',
    metadata={
      'description': f'{MODEL} evaluation for GDGPT'
    }
  )
  print(f"Created batch with ID: {batch.id}")

def check_batch():
  assert BATCH_ID is not None, "Batch ID not provided."
  batch = client.batches.retrieve(BATCH_ID)
  print(f"Batch status: {batch.status}")
  print(batch)
  # print(f"Batch status: {batch.status}")
  # print(f"Batch completion count: {batch.completion_count}")
  
def cancel_batch():
  assert BATCH_ID is not None, "Batch ID not provided."
  client.batches.cancel(BATCH_ID)
  print(f"Cancelled batch with ID: {BATCH_ID}")
  
def parse_batch():
  assert BATCH_ID is not None, "Batch ID not provided."
  output_file_id = client.batches.retrieve(BATCH_ID).output_file_id
  
  output_text = client.files.content(output_file_id).text
  
  batch_output = [json.loads(line) for line in output_text.split('\n') if line]
  
  print(batch_output[0])
  
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  with open(f'{OUTPUT_DIR}/{FILE_NAME}_output.jsonl', 'w') as f:
    f.write(output_text)
      
  print(f"Saved completions to {OUTPUT_DIR}/{FILE_NAME}_output.jsonl")
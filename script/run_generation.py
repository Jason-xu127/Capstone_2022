from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small-local")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small-local")

filename = 'data/test_short.json'

output = 'predictions.txt'

f2 = open(output,'w')

with open(filename) as f1:
    data = json.load(f1)


for idx, v in enumerate(data):
    each = data[v]
    content = each['content']

    for dialogue in content:

        context = dialogue['context']

        new_user_input_ids = tokenizer.encode(context + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

        f2.write(response+'\n')
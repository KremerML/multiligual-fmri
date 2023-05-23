import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from nltk.tokenize import word_tokenize
import string
import re

# Load pre-trained model tokenizer (vocabulary)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Load pre-trained model (weights)
model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
model = model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    return tokens

if __name__ == '__main__':

    
    # Read and preprocess text file
    with open('./LM_analysis/le_petit_prince_full.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = preprocess_text(text)

    # Chunking tokens into manageable sizes
    MAX_LEN = 510
    input_chunks = [tokens[i:i+MAX_LEN] for i in range(0, len(tokens), MAX_LEN)]

    # Convert each chunk to input ids and get hidden states
    last_hidden_states = []
    for chunk in input_chunks:
        input_ids_chunk = torch.tensor([tokenizer.encode(chunk, add_special_tokens=True)]).to(device)
        with torch.no_grad():
            outputs = model(input_ids_chunk)
            last_hidden_states.append(outputs[0])

    # Save the last hidden states
    torch.save(last_hidden_states, 'last_hidden_states.pt')
    print('Last hidden states saved to last_hidden_states.pt')

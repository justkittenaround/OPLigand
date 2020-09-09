import torch
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, receptors, ligands, device, max_length=50):
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    tokens = [chr(i+65) for i in range(26)]
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, receptors.init_token)
    tokens.append(receptors.eos_token)
    # Go through each german token and convert to an index
    text_to_indices = [receptors.vocab.stoi[token] for token in tokens]
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    # Build encoder hidden, cell state
    outputs = [ligands.vocab.stoi["*"]]
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)
        best_guess = output.argmax(2)[-1,:].item()
        outputs.append(best_guess)
        if best_guess == ligands.vocab.stoi["!"]:
            break
    translated_sentence = [ligands.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]



# score = bleu(test_data[1:100], model, receptors, ligands, device)
def bleu(data, model, receptors, ligands, device):
    print("=> Calculating Bleu")
    targets = []
    outputs = []
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        prediction = translate_sentence(model, src, receptors, ligands, device)
        prediction = list(map(str, prediction[:-1]))  # remove <eos> token
        outputs.append(prediction)
        targets.append(list(map(str, trg)))
    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

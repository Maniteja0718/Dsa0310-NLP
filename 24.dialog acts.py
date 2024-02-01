import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('nps_chat')

def dialog_act_recognition(dialog):
    dialog_acts = []
    for utterance in dialog:
        words = word_tokenize(utterance)
        dialog_act = nltk.pos_tag(words)
        dialog_acts.append(dialog_act)
    return dialog_acts

if __name__ == "__main__":
    # Example conversation
    conversation = [
        "Hey, how's it going?",
        "Not too bad, thanks for asking.",
       
    ]

    dialog_acts = dialog_act_recognition(conversation)
    for i, utterance in enumerate(dialog_acts):
        print(f"Utterance {i+1}: {utterance}")

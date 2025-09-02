import torch
from transformers import BertForTokenClassification, AutoTokenizer

class NER:
    def __init__(self, model_path: str = "model/model_korets.pth", use_cuda: bool = False):
        self.label2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-DATE': 3, 'I-DATE': 4}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = BertForTokenClassification.from_pretrained(
            "prajjwal1/bert-tiny",
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.id2label[p.item()] for p in predictions[0]]

        return list(zip(tokens, labels))

    def anonymize(self, text):
        res = self.predict(text)

        mapping = {
        'B-DATE': '{DATE}', 'I-DATE': '{DATE}',
        'B-LOC': '{LOCATION}', 'I-LOC': '{LOCATION}',
        }

        fin = []
        last_mask = None

        for tok, *_, label in res[1:-1]:
            piece = mapping.get(label, tok)

            if isinstance(piece, str) and piece.startswith("##") and fin:
                fin[-1] += piece[2:]
                continue

            if piece in ('{DATE}', '{LOCATION}'):
                if last_mask == piece:
                    continue
                last_mask = piece
            else:
                last_mask = None

            fin.append(piece)

        return " ".join(fin)
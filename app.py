import os
import re
import json
import unicodedata
import numpy as np
from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ── App setup ─────────────────────────────────────────────────────
app = Flask(__name__)

# ── Constants (mirror your notebook exactly) ──────────────────────
HF_REPO_ID  = "Bhupendraencode/multilingual"
MAX_LENGTH  = 160
THRESHOLD   = 0.50   # overridden by inference_config.json if present
ID2LABEL    = {0: "FAKE", 1: "REAL"}

CLICKBAIT_WORDS = {
    'shocking', 'unbelievable', "you won't believe", 'mind-blowing',
    'secret', "they don't want you", 'breaking', 'exposed', 'truth about',
    'doctors hate', 'one weird trick', 'miracle', 'conspiracy',
}
HEDGE_WORDS = {
    'allegedly', 'reportedly', 'claimed', 'unverified', 'rumored',
    'sources say', 'according to', 'suggest', 'may have', 'could be',
}

_URL_RE        = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_HTML_RE       = re.compile(r'<[^>]+>')
_MULTI_WS      = re.compile(r'\s+')
_QUOTE_MARKS   = re.compile(r'[\"\"\"\'\'\"«»]')
_EMOJ_RE       = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
_REPEATED_PUNCT = re.compile(r'([!?.,;:]){2,}')
_HASHTAG_RE     = re.compile(r'#\w+')
_MENTION_RE     = re.compile(r'@\w+')


# ── Text utilities (exact copy from notebook) ─────────────────────
def clean_text(text) -> str:
    if text is None:
        return '[EMPTY]'
    text = str(text)
    text = _URL_RE.sub(' ', text)
    text = _HTML_RE.sub(' ', text)
    text = _HASHTAG_RE.sub(' ', text)
    text = _MENTION_RE.sub(' ', text)
    text = _EMOJ_RE.sub(' ', text)
    text = unicodedata.normalize('NFC', text)
    text = _QUOTE_MARKS.sub('"', text)
    text = _REPEATED_PUNCT.sub(r'\1', text)
    text = _MULTI_WS.sub(' ', text).strip()
    return text if len(text) >= 2 else '[EMPTY]'


def extract_text_features(text: str) -> dict:
    text_lower = text.lower()
    words      = text_lower.split()
    n_words    = max(len(words), 1)
    n_caps     = sum(1 for c in text if c.isupper())
    n_exclaim  = text.count('!')
    n_digits   = sum(1 for c in text if c.isdigit())
    clickbait_count = sum(1 for w in CLICKBAIT_WORDS if w in text_lower)
    hedge_count     = sum(1 for w in HEDGE_WORDS     if w in text_lower)
    caps_ratio = n_caps / max(len(text), 1)
    return {
        'word_count':      n_words,
        'caps_ratio':      round(caps_ratio, 3),
        'exclaim_count':   n_exclaim,
        'digit_ratio':     round(n_digits / max(len(text), 1), 3),
        'clickbait_count': clickbait_count,
        'hedge_count':     hedge_count,
        'avg_word_len':    round(np.mean([len(w) for w in words]) if words else 0, 2),
    }


def build_enriched_text(cleaned: str, feats: dict, lang: str = 'en') -> str:
    signals = []
    if feats['clickbait_count'] > 0:
        signals.append(f'clickbait:{feats["clickbait_count"]}')
    if feats['caps_ratio'] > 0.15:
        signals.append('high_caps')
    if feats['exclaim_count'] >= 2:
        signals.append('multi_exclaim')
    if feats['hedge_count'] > 0:
        signals.append(f'hedged:{feats["hedge_count"]}')
    signals.append(f'lang:{lang}')
    return cleaned + (' [SEP] ' + ' '.join(signals) if signals else '')


# ── Model architecture (exact copy from notebook Cell 17) ─────────
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attn(hidden_states).squeeze(-1)
        mask   = (1 - attention_mask) * -1e9
        scores = torch.softmax(scores + mask, dim=-1)
        return torch.bmm(scores.unsqueeze(1), hidden_states).squeeze(1)


class FakeNewsClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, dropout: float):
        super().__init__()
        config        = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=config)
        hidden_size   = config.hidden_size
        self.pool       = AttentionPooling(hidden_size)
        self.dropout1   = nn.Dropout(dropout)
        self.dense1     = nn.Linear(hidden_size, hidden_size // 2)
        self.act        = nn.GELU()
        self.dropout2   = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size // 2, num_labels)

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        pooled = self.pool(hidden, attention_mask)
        x      = self.dropout1(pooled)
        x      = self.dense1(x)
        x      = self.act(x)
        x      = self.dropout2(x)
        return self.classifier(x)


# ── Model loading (runs once at startup) ──────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = None
model     = None
threshold = THRESHOLD


def load_model():
    global tokenizer, model, threshold

    print(f"Loading model from {HF_REPO_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, use_fast=True)

    # Load inference config if available
    try:
        from huggingface_hub import hf_hub_download
        cfg_path  = hf_hub_download(HF_REPO_ID, "inference_config.json")
        with open(cfg_path) as f:
            cfg_inf = json.load(f)
        threshold = float(cfg_inf.get("decision_threshold", THRESHOLD))
        dropout   = float(cfg_inf.get("dropout", 0.25))
        print(f"  threshold = {threshold}  dropout = {dropout}")
    except Exception as e:
        print(f"  inference_config.json not found ({e}), using defaults")
        dropout = 0.25

    model = FakeNewsClassifier(
        backbone_name=HF_REPO_ID,
        num_labels=2,
        dropout=dropout,
    )

    # Load custom head weights
    try:
        from huggingface_hub import hf_hub_download
        head_path = hf_hub_download(HF_REPO_ID, "classification_head.pt")
        head_sd   = torch.load(head_path, map_location='cpu', weights_only=True)
        model.pool.load_state_dict(head_sd['pool'])
        model.dense1.load_state_dict(head_sd['dense1'])
        model.classifier.load_state_dict(head_sd['classifier'])
        if 'dropout1' in head_sd:
            model.dropout1.load_state_dict(head_sd['dropout1'])
        if 'dropout2' in head_sd:
            model.dropout2.load_state_dict(head_sd['dropout2'])
        print("  Custom head weights loaded")
    except Exception as e:
        print(f"  Could not load classification_head.pt ({e}). Using random head.")

    model.to(device)
    model.eval()
    print(f"Model ready on {device}")


# ── Inference ──────────────────────────────────────────────────────
def predict(text: str, lang: str = 'en') -> dict:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'label': 'INVALID', 'confidence': 0.0,
                'fake_prob': 0.0, 'real_prob': 0.0, 'features': {}}

    cleaned  = clean_text(text)
    feats    = extract_text_features(cleaned)
    enriched = build_enriched_text(cleaned, feats, lang)

    enc = tokenizer(
        enriched,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
        probs  = torch.softmax(logits.float(), dim=-1).squeeze().cpu().numpy()

    fake_p   = float(probs[0])
    real_p   = float(probs[1])
    pred_idx = int(real_p >= threshold)
    label    = ID2LABEL[pred_idx]

    return {
        'label':      label,
        'confidence': round(float(probs[pred_idx]) * 100, 1),
        'fake_prob':  round(fake_p * 100, 1),
        'real_prob':  round(real_p * 100, 1),
        'threshold':  round(threshold, 2),
        'features':   feats,
        'cleaned':    cleaned,
    }


# ── Routes ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en').strip() or 'en'

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 chars)'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded yet, please retry in a moment'}), 503

    result = predict(text, lang)
    return jsonify(result)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


# ── Entry point ────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # Called by gunicorn
    load_model()

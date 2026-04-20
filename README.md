# Multilingual Fake News Detector — Flask + Heroku

## Project Structure

```
fakenews_app/
├── app.py               ← Flask backend + inference pipeline
├── templates/
│   └── index.html       ← Full HTML/CSS/JS frontend
├── requirements.txt     ← Python dependencies
├── Procfile             ← Heroku process definition
├── runtime.txt          ← Python version pin
└── .gitignore
```

---

## Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
python app.py
# → Open http://localhost:5000
```

---

## Deploy to Heroku — Step by Step

### Step 1 — Install Heroku CLI
Download from https://devcenter.heroku.com/articles/heroku-cli
Verify: `heroku --version`

### Step 2 — Login to Heroku
```bash
heroku login
```

### Step 3 — Initialise Git and create Heroku app
```bash
cd fakenews_app

git init
git add .
git commit -m "Initial commit — multilingual fake news detector"

heroku create fakenews-detector
# Or pick your own name:
# heroku create your-app-name
```

### Step 4 — Set the Python stack
```bash
heroku buildpacks:set heroku/python
```

### Step 5 — Increase Heroku slug size limit (PyTorch is large)
PyTorch + Transformers exceeds the default 500MB slug.
Use the Heroku-22 stack which has a 1GB limit:
```bash
heroku stack:set heroku-22
```

### Step 6 — Deploy
```bash
git push heroku main
```
First deploy takes 5–10 minutes (downloading PyTorch).

### Step 7 — Open your app
```bash
heroku open
```
Your URL: `https://fakenews-detector.herokuapp.com`

### Step 8 — Check logs if anything fails
```bash
heroku logs --tail
```

---

## Important Heroku Notes

### Dyno timeout
Model loading from Hugging Face Hub takes ~60–90 seconds on first request.
The Procfile sets `--timeout 120` to handle this.

### Free tier vs Eco dynos
Heroku's free tier was discontinued. Use **Eco dynos** ($5/month).
```bash
heroku ps:scale web=1
heroku dyno:type web=eco
```

### Slug size
If you hit slug size limits, add to `.slugignore`:
```
*.pt
*.bin
venv/
```
The model is loaded from Hugging Face Hub at runtime, not bundled in the slug.

### Environment variables (optional)
```bash
heroku config:set HF_TOKEN=your_token_here   # if repo is private
```

---

## How the App Works

1. User types a claim and selects a language
2. Frontend sends POST `/predict` with `{text, lang}`
3. `app.py` runs: clean_text → extract_text_features → build_enriched_text → tokenize → model forward pass → softmax
4. Returns `{label, fake_prob, real_prob, confidence, features}`
5. Frontend renders verdict, probability bars, signal pills, stats grid

The inference pipeline is an exact copy of the notebook's `predict()` function.

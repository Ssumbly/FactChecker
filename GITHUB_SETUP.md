# GitHub Upload Instructions

## ✅ Pre-Upload Checklist

All sensitive files are properly excluded:
- ✅ `.env` (API key) - **IGNORED**
- ✅ `*.wav`, `*.mp3` (audio files) - **IGNORED**
- ✅ `*_cache.json` (cache files) - **IGNORED**
- ✅ `venv/` (virtual environment) - **IGNORED**
- ✅ `uploads/` (user uploads) - **IGNORED**
- ✅ `__pycache__/` (Python cache) - **IGNORED**

## Files Ready to Commit

Safe files that will be uploaded:
- `.env.example` - Template for environment variables (no real keys)
- `.gitignore` - Git ignore rules
- `COST_ANALYSIS.md` - Cost documentation
- `README.md` - Project documentation
- `app.py` - Flask web application
- `main.py` - CLI application
- `requirements.txt` - Python dependencies
- `templates/index.html` - Web interface

## Next Steps

1. **Review what will be committed:**
   ```bash
   git status
   ```

2. **Create your first commit:**
   ```bash
   git commit -m "Initial commit: AI-Powered Fact Checker"
   ```

3. **Create a GitHub repository:**
   - Go to https://github.com/new
   - Create a new repository
   - **DO NOT** initialize with README (you already have one)
   - Copy the repository URL

4. **Connect and push:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## After Uploading

Tell users to:
1. Clone the repository
2. Copy `.env.example` to `.env`
3. Add their API key to `.env`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `python main.py` or `python app.py`

## Security Reminders

- ✅ Never commit `.env` file
- ✅ Never commit API keys
- ✅ Use `.env.example` as a template only
- ✅ `.gitignore` is configured correctly


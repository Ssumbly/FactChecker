

An AI-powered fact-checking application built with Python that uses Anthropic's Claude API to verify facts and claims from audio input.


## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fact-checker.git
   cd fact-checker
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key:**
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env  # Windows
     cp .env.example .env    # macOS/Linux
     ```
   - Open `.env` and replace `your_api_key_here` with your actual Anthropic API key
   - Get your API key from: https://console.anthropic.com/

## Usage

### Command Line Interface

Run the application:
```bash
python main.py
```

This will:
- Record 10 seconds of audio (or use existing `recording.wav`)
- Transcribe the audio using Google Speech Recognition
- Extract claims from the transcript
- Fact-check each claim in parallel
- Display results in the terminal with color-coded output
- Save results to a timestamped JSON file

### Web Application (Flask)

**Start the Flask server:**
```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

**Access the web interface:**
1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. You'll see a clean web interface where you can:
   - **Record audio** directly in your browser (click "Record Audio")
   - **Upload an audio file** (click "Choose File")
   - **Process and fact-check** (click "Upload & Check")

**Web App Features:**
- Record audio using your browser's microphone
- Upload audio files (supports: wav, mp3, m4a, ogg, webm)
- View transcript and extracted claims
- See fact-check results with color-coded verdicts (True/False/Unverifiable)
- Results displayed in clean, minimal cards

**Note:** The web app automatically converts browser recordings (WebM) to WAV format for processing.

## Project Structure

```
fact-checker/
├── main.py              # CLI application entry point
├── app.py               # Flask web application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys) - NOT in git
├── .env.example         # Template for environment variables
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
├── templates/
│   └── index.html      # Web interface template
└── COST_ANALYSIS.md    # API cost analysis documentation
```

## Cost Optimization

The application is optimized for low API costs:
- Uses Claude Haiku for transcription/extraction (10x cheaper)
- Uses Claude Haiku for fact-checking
- Implements caching to avoid duplicate API calls
- Estimated cost: ~$0.02-0.03 per request (first time), $0.00 for cached results

See `COST_ANALYSIS.md` for detailed cost breakdown.

## Requirements

- Python 3.7+
- Anthropic API key
- Internet connection for API calls
- Microphone (for recording) or audio files (for upload)

## License

This project is open source and available for modification.

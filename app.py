"""
Flask web application for AI-Powered Fact Checker
"""

import os
import time
import asyncio
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import dotenv
from anthropic import Anthropic

# Import functions from main.py
from main import (
    transcribe_audio,
    extract_claims,
    fact_check_claim_async,
    load_cache,
    get_claim_hash,
    get_cached_result
)

# Load environment variables
dotenv.load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.urandom(24)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'webm', 'mpeg'}

# Initialize Anthropic client
api_key = os.getenv("ANTHROPIC_API_KEY")
client = None
if api_key:
    client = Anthropic(api_key=api_key)
    # Load cache on startup
    load_cache()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio file upload and process fact-checking"""
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: wav, mp3, m4a, ogg, webm'}), 400
        
        # Check API key
        if not client:
            return jsonify({'error': 'ANTHROPIC_API_KEY not configured'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        # Handle webm files from browser MediaRecorder - convert or use as-is
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'webm'
        temp_filename = f'temp_audio_{int(time.time())}.{file_ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(filepath)
        
        # Step 1: Transcribe audio
        # transcribe_audio can handle webm/ogg files from MediaRecorder
        transcribed_text = transcribe_audio(filepath, client)
        if not transcribed_text:
            return jsonify({'error': 'Transcription failed'}), 500
        
        # Step 2: Extract claims
        claims = extract_claims(transcribed_text, client)
        if not claims:
            return jsonify({
                'transcript': transcribed_text,
                'claims': [],
                'results': [],
                'message': 'No factual claims found in the transcript'
            }), 200
        
        # Step 3: Fact-check claims in parallel
        async def fact_check_all():
            tasks = [
                fact_check_claim_async(claim, client, use_cache=True)
                for claim in claims
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run async fact-checking
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        results = asyncio.run(fact_check_all())
        
        # Process results
        fact_check_results = []
        for claim, result in zip(claims, results):
            if isinstance(result, Exception):
                fact_check_results.append({
                    'claim': claim,
                    'result': None,
                    'error': str(result)
                })
            elif result:
                fact_check_results.append({
                    'claim': claim,
                    'result': result
                })
            else:
                fact_check_results.append({
                    'claim': claim,
                    'result': None,
                    'error': 'Fact-check failed'
                })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Return results
        return jsonify({
            'transcript': transcribed_text,
            'claims': claims,
            'results': fact_check_results,
            'message': 'Processing completed successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


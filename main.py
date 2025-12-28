
"""
AI-Powered Fact Checker
Main entry point for the fact-checker application.
"""

import os
import base64
import json
import re
import time
import wave
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
import dotenv
import pyaudio
import speech_recognition as sr
from pydub import AudioSegment
from anthropic import Anthropic

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    # Reset
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

def print_boxed_text(title, text, width=70):
    """Print text in a bordered box with a title"""
    lines = text.split('\n')
    # Wrap long lines
    wrapped_lines = []
    for line in lines:
        if len(line) <= width - 4:
            wrapped_lines.append(line)
        else:
            words = line.split(' ')
            current_line = ""
            for word in words:
                if len(current_line + word + " ") <= width - 4:
                    current_line += word + " "
                else:
                    if current_line:
                        wrapped_lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                wrapped_lines.append(current_line.strip())
    
    # Print box
    border = "═" * (width - 2)
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔═{border}═╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║ {Colors.BRIGHT_WHITE}{title.center(width - 2)}{Colors.CYAN} ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╠═{border}═╣{Colors.RESET}")
    
    for line in wrapped_lines:
        print(f"{Colors.BOLD}{Colors.CYAN}║ {Colors.RESET}{line.ljust(width - 2)}{Colors.BOLD}{Colors.CYAN} ║{Colors.RESET}")
    
    print(f"{Colors.BOLD}{Colors.CYAN}╚═{border}═╝{Colors.RESET}\n")

def format_url(url):
    """Format a string as a clickable-looking URL"""
    # Add https:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return f"{Colors.BRIGHT_CYAN}{url}{Colors.RESET}"

def get_verdict_color(verdict, confidence):
    """Get color code based on verdict and confidence"""
    verdict_upper = verdict.upper()
    
    if verdict_upper == "TRUE" and confidence > 70:
        return Colors.BRIGHT_GREEN
    elif verdict_upper == "FALSE" and confidence > 70:
        return Colors.BRIGHT_RED
    elif verdict_upper == "UNVERIFIABLE":
        return Colors.BRIGHT_YELLOW
    else:
        return Colors.YELLOW  # Lower confidence cases

def get_verdict_icon(verdict):
    """Get emoji/symbol for verdict"""
    verdict_upper = verdict.upper()
    if verdict_upper == "TRUE":
        return "✓"
    elif verdict_upper == "FALSE":
        return "✗"
    else:
        return "?"

# Load environment variables
dotenv.load_dotenv()

def record_audio(duration=10, filename="recording.wav"):
    """
    Record audio from the microphone for a specified duration.
    
    Args:
        duration (int): Recording duration in seconds (default: 10)
        filename (str): Name of the output WAV file (default: "recording.wav")
    
    Returns:
        str: Filepath of the saved audio file, or None if recording failed
    """
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    audio = None
    stream = None
    
    try:
        print(f"Starting audio recording for {duration} seconds...")
        print("Recording in progress...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Open audio stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print(f"Recording... Speak now!")
        
        # Record audio in chunks
        frames = []
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        print("Recording finished!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save recorded audio to WAV file
        print(f"Saving audio to {filename}...")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        filepath = os.path.abspath(filename)
        print(f"Audio saved successfully to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error during audio recording: {str(e)}")
        
        # Clean up resources in case of error
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        
        if audio is not None:
            try:
                audio.terminate()
            except:
                pass
        
        return None

def transcribe_audio(audio_filepath, client=None, use_cache=True):
    """
    Transcribe audio file using speech recognition library (Google API).
    
    Args:
        audio_filepath (str): Path to the WAV audio file
        client (Anthropic): Anthropic API client instance (optional, not used for transcription)
        use_cache (bool): Whether to use cached transcriptions (default: True)
    
    Returns:
        str: Transcribed text, or None if transcription failed
    """
    try:
        # Validate file exists
        if not os.path.exists(audio_filepath):
            print(f"Error: Audio file not found: {audio_filepath}")
            return None
        
        # Check cache first (hash of audio file)
        if use_cache:
            with open(audio_filepath, 'rb') as f:
                audio_hash = hashlib.md5(f.read()).hexdigest()
            
            if audio_hash in transcript_cache:
                print(f"Using cached transcription for audio file")
                return transcript_cache[audio_hash]['transcript']
        
        print(f"Reading audio file: {audio_filepath}")
        
        # Convert audio file to WAV if needed (speech_recognition requires WAV)
        file_ext = audio_filepath.rsplit('.', 1)[1].lower() if '.' in audio_filepath else ''
        wav_filepath = audio_filepath
        
        if file_ext not in ['wav']:
            print(f"Converting {file_ext} to WAV format...")
            try:
                # Load audio with pydub (requires ffmpeg for webm/ogg)
                audio_segment = AudioSegment.from_file(audio_filepath, format=file_ext)
                # Convert to WAV
                wav_filepath = audio_filepath.rsplit('.', 1)[0] + '.wav'
                audio_segment.export(wav_filepath, format="wav")
                print(f"Conversion complete: {wav_filepath}")
            except Exception as e:
                error_msg = str(e)
                print(f"Error converting audio: {error_msg}")
                if "ffmpeg" in error_msg.lower() or "avconv" in error_msg.lower():
                    print("Note: Audio conversion requires ffmpeg. Please install ffmpeg:")
                    print("  Windows: Download from https://ffmpeg.org/download.html")
                    print("  Or use WAV files directly for recording")
                return None
        
        # Use speech_recognition library for transcription (free Google API)
        print("Transcribing audio using speech recognition...")
        recognizer = sr.Recognizer()
        
        # Read the audio file
        with sr.AudioFile(wav_filepath) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Read the entire audio file
            audio = recognizer.record(source)
        
        # Transcribe using Google's free speech recognition API
        try:
            transcribed_text = recognizer.recognize_google(audio)
            print(f"Transcription completed: {len(transcribed_text)} characters")
        except sr.UnknownValueError:
            print("Error: Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Error: Could not request results from speech recognition service: {str(e)}")
            # Fallback: try with offline recognition
            try:
                print("Trying offline recognition...")
                transcribed_text = recognizer.recognize_sphinx(audio)
            except:
                print("Offline recognition also failed. Please check your internet connection.")
                return None
        
        if transcribed_text:
            transcribed_text = transcribed_text.strip()
            # Cache the transcription (use original file for hash)
            if use_cache:
                with open(audio_filepath, 'rb') as f:
                    audio_hash = hashlib.md5(f.read()).hexdigest()
                transcript_cache[audio_hash] = {
                    'transcript': transcribed_text,
                    'timestamp': datetime.now().isoformat()
                }
                # Save cache
                try:
                    with open(TRANSCRIPT_CACHE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(transcript_cache, f, indent=2, ensure_ascii=False)
                except:
                    pass
            
            # Clean up converted WAV file if it was created
            if wav_filepath != audio_filepath and os.path.exists(wav_filepath):
                try:
                    os.remove(wav_filepath)
                except:
                    pass
            
            print("[SUCCESS] Audio transcription completed!")
            return transcribed_text
        else:
            print("Warning: Received empty transcription from API")
            return None
            
    except FileNotFoundError:
        print(f"Error: Audio file not found: {audio_filepath}")
        return None
    except Exception as e:
        error_msg = str(e)
        print(f"Error during audio transcription: {error_msg}")
        
        # Provide more specific error messages
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            print("  Hint: Check that ANTHROPIC_API_KEY is correctly set in .env file")
        elif "rate limit" in error_msg.lower():
            print("  Hint: Rate limit exceeded. Please wait before trying again.")
        elif "invalid" in error_msg.lower() or "format" in error_msg.lower():
            print("  Hint: Audio file format may not be supported")
        
        return None

def extract_claims(transcript, client=None):
    """
    Extract factual claims from a transcript using Claude API.
    
    Args:
        transcript (str): The transcribed text to extract claims from
        client (Anthropic): Anthropic API client instance (optional)
    
    Returns:
        list: List of extracted claim strings, or empty list if extraction fails
    """
    # Initialize client if not provided
    if client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found in .env file")
            return []
        client = Anthropic(api_key=api_key)
    
    prompt = f"""Extract ALL statements from the following transcript that can be fact-checked or verified. This includes:
- Factual statements about people, places, events, statistics
- Claims about groups of people (ethnicities, nationalities, religions, etc.)
- Generalizations and stereotypes that can be verified
- Statements about cause and effect
- Claims about characteristics or behaviors
- Any assertion that makes a verifiable claim about reality

DO NOT ignore opinions, stereotypes, or generalizations - these are claims that can and should be fact-checked.
ONLY ignore: questions, greetings, conversational fillers like "um" or "uh", pure expressions of emotion without claims.

Transcript:
{transcript}

Return ONLY a JSON array of claim strings. Each claim should be a standalone statement that can be fact-checked. Format:
["claim1", "claim2", "claim3"]

If no verifiable claims are found, return an empty array: []"""
    
    try:
        print("Extracting claims from transcript...")
        
        # Using Haiku for claim extraction (10x cheaper than Sonnet)
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Much cheaper: $0.80/$4 vs $3/$15 per million tokens
            max_tokens=1024,  # Reduced from 2048 (sufficient for claim extraction)
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract response text
        response_text = ""
        if message.content:
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text
                elif isinstance(block, dict) and block.get('type') == 'text':
                    response_text += block.get('text', '')
        
        # Try to parse JSON array
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            claims = json.loads(json_str)
            
            if isinstance(claims, list):
                # Filter out empty strings and validate
                claims = [claim.strip() for claim in claims if claim and claim.strip()]
                print(f"  Extracted {len(claims)} claim(s)")
                return claims
        
        print("Warning: Could not parse claims from response")
        return []
        
    except json.JSONDecodeError as e:
        print(f"Error parsing claims JSON: {str(e)}")
        return []
    except Exception as e:
        print(f"Error extracting claims: {str(e)}")
        return []

# Cache for fact-check results
CACHE_FILE = "fact_check_cache.json"
fact_check_cache = {}

# Cache for transcriptions (same audio = same transcript)
TRANSCRIPT_CACHE_FILE = "transcript_cache.json"
transcript_cache = {}

def load_cache():
    """Load fact-check and transcript caches from file"""
    global fact_check_cache, transcript_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                fact_check_cache = json.load(f)
        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not load cache: {str(e)}{Colors.RESET}")
            fact_check_cache = {}
    
    if os.path.exists(TRANSCRIPT_CACHE_FILE):
        try:
            with open(TRANSCRIPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                transcript_cache = json.load(f)
        except Exception as e:
            transcript_cache = {}
    
    if os.path.exists(TRANSCRIPT_CACHE_FILE):
        try:
            with open(TRANSCRIPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                transcript_cache = json.load(f)
        except Exception as e:
            transcript_cache = {}

def save_cache():
    """Save fact-check cache to file"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(fact_check_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not save cache: {str(e)}{Colors.RESET}")

def get_claim_hash(claim):
    """Generate a hash for a claim to use as cache key"""
    return hashlib.md5(claim.lower().strip().encode('utf-8')).hexdigest()

def get_cached_result(claim):
    """Check if claim result is cached"""
    claim_hash = get_claim_hash(claim)
    return fact_check_cache.get(claim_hash)

def cache_result(claim, result):
    """Cache a fact-check result"""
    claim_hash = get_claim_hash(claim)
    fact_check_cache[claim_hash] = {
        'claim': claim,
        'result': result,
        'timestamp': datetime.now().isoformat()
    }
    save_cache()

def fact_check_claim(claim, client=None, max_retries=3, retry_delay=1, use_cache=True):
    """
    Fact-check a claim using Claude API with web search.
    
    Args:
        claim (str): The claim to fact-check
        client (Anthropic): Anthropic API client instance (optional)
        max_retries (int): Maximum number of retry attempts (default: 3)
        retry_delay (float): Delay between retries in seconds (default: 1)
        use_cache (bool): Whether to use cached results (default: True)
    
    Returns:
        dict: Dictionary with 'verdict', 'confidence', 'explanation', 'sources'
              Returns None if fact-checking fails
    """
    # Check cache first
    if use_cache:
        cached = get_cached_result(claim)
        if cached and 'result' in cached:
            return cached['result']
    
    # Initialize client if not provided
    if client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found in .env file")
            return None
        client = Anthropic(api_key=api_key)
    
    prompt = f"""Fact-check this claim using web search: {claim}

Provide: verdict (true/false/unverifiable), confidence (0-100), brief explanation, and sources used. Return as JSON in the following format:
{{
    "verdict": "true|false|unverifiable",
    "confidence": <number 0-100>,
    "explanation": "<brief explanation>",
    "sources": ["<source1>", "<source2>", ...]
}}"""
    
    for attempt in range(max_retries):
        try:
            print(f"Fact-checking claim: '{claim}' (attempt {attempt + 1}/{max_retries})...")
            
            # Create message for fact-checking (using Sonnet for accuracy, but simplified)
            # Removed tool_use loop to prevent extra API calls - Claude's knowledge is sufficient
            message = client.messages.create(
                model="claude-3-haiku-20240307",  # Using Haiku (cheaper and available) - Sonnet models not accessible
                max_tokens=2048,  # Reduced from 4096 (sufficient for fact-check responses)
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract response text
            response_text = ""
            if message.content:
                for block in message.content:
                    if hasattr(block, 'text'):
                        response_text += block.text
                    elif isinstance(block, dict) and block.get('type') == 'text':
                        response_text += block.get('text', '')
            
            if not response_text:
                raise ValueError("Empty response from API")
            
            # Try to parse JSON from response
            # Look for JSON block in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['verdict', 'confidence', 'explanation', 'sources']
                if all(field in result for field in required_fields):
                    fact_check_result = {
                        'verdict': result['verdict'],
                        'confidence': int(result['confidence']),
                        'explanation': result['explanation'],
                        'sources': result['sources'] if isinstance(result['sources'], list) else [result['sources']]
                    }
                    # Cache the result
                    if use_cache:
                        cache_result(claim, fact_check_result)
                    return fact_check_result
            
            # If JSON parsing failed, try to extract from text
            print("Warning: Could not parse JSON from response")
            print(f"  Full response: {response_text[:300]}...")
            
            # Try to extract verdict from text
            verdict = 'unverifiable'
            if 'true' in response_text.lower()[:200]:
                verdict = 'true'
            elif 'false' in response_text.lower()[:200]:
                verdict = 'false'
            
            # Fallback: return structured data from text analysis
            return {
                'verdict': verdict,
                'confidence': 50,
                'explanation': response_text[:500] if len(response_text) > 0 else "No explanation provided",
                'sources': []
            }
            
        except json.JSONDecodeError as e:
            print(f"  JSON parsing error: {str(e)}")
            print(f"  Response text: {response_text[:200]}...")  # Show first 200 chars
            if attempt < max_retries - 1:
                print(f"  Retrying...")
                time.sleep(retry_delay)
                continue
            else:
                # Return fallback result instead of None
                print("  Using fallback result format")
                return {
                    'verdict': 'unverifiable',
                    'confidence': 50,
                    'explanation': f"Could not parse structured response: {response_text[:300]}",
                    'sources': []
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"  Error during fact-checking: {error_msg}")
            
            # Provide specific error messages
            if "rate limit" in error_msg.lower():
                retry_delay = min(retry_delay * 2, 10)  # Exponential backoff, max 10 seconds
                if attempt < max_retries - 1:
                    print(f"  Rate limited. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                print("  Authentication error: Check your ANTHROPIC_API_KEY")
                return None
            elif "invalid_request_error" in error_msg.lower():
                print(f"  Invalid request: {error_msg}")
                # Don't retry on invalid requests
                return None
            
            # For other errors, retry if attempts remain
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10)  # Exponential backoff
            else:
                print(f"  Failed after {max_retries} attempts")
                return None
    
    return None

async def fact_check_claim_async(claim, client=None, max_retries=3, retry_delay=1, use_cache=True, progress_callback=None):
    """
    Async wrapper for fact_check_claim to enable parallel processing.
    
    Args:
        claim (str): The claim to fact-check
        client (Anthropic): Anthropic API client instance (optional)
        max_retries (int): Maximum number of retry attempts (default: 3)
        retry_delay (float): Delay between retries in seconds (default: 1)
        use_cache (bool): Whether to use cached results (default: True)
        progress_callback: Optional callback function for progress updates
    
    Returns:
        dict: Dictionary with 'verdict', 'confidence', 'explanation', 'sources'
              Returns None if fact-checking fails
    """
    # Run the sync function in a thread pool
    loop = asyncio.get_event_loop()
    
    # Check cache first (fast, no need for async)
    if use_cache:
        cached = get_cached_result(claim)
        if cached and 'result' in cached:
            if progress_callback:
                progress_callback(claim, "cached")
            return cached['result']
    
    if progress_callback:
        progress_callback(claim, "checking")
    
    # Run the blocking fact_check_claim in a thread
    result = await loop.run_in_executor(
        None,
        lambda: fact_check_claim(claim, client, max_retries, retry_delay, use_cache)
    )
    
    if progress_callback:
        progress_callback(claim, "completed" if result else "failed")
    
    return result

class ProgressTracker:
    """Track progress of parallel fact-checking operations"""
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.cached = 0
        self.status = {}  # Track status of each claim
    
    def update(self, claim, status):
        """Update progress for a claim"""
        if status == "cached":
            self.cached += 1
        elif status == "completed":
            self.completed += 1
        elif status == "failed":
            self.failed += 1
        
        self.status[claim] = status
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress"""
        total_done = self.completed + self.failed + self.cached
        progress_pct = (total_done / self.total * 100) if self.total > 0 else 0
        bar_length = 40
        filled = int(bar_length * total_done / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Print progress bar
        print(f"\r{Colors.BRIGHT_CYAN}Progress:{Colors.RESET} [{bar}] {progress_pct:.1f}% | "
              f"{Colors.BRIGHT_GREEN}✓ {self.completed}{Colors.RESET} | "
              f"{Colors.YELLOW}💾 {self.cached}{Colors.RESET} | "
              f"{Colors.BRIGHT_RED}✗ {self.failed}{Colors.RESET} | "
              f"{Colors.WHITE}{total_done}/{self.total}{Colors.RESET}", end='', flush=True)

def save_results_to_json(transcript, claims, fact_check_results, filename=None):
    """Save fact-check results to a JSON file with timestamp"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fact_check_results_{timestamp}.json"
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'transcript': transcript,
        'claims': claims,
        'results': []
    }
    
    for item in fact_check_results:
        result_entry = {
            'claim': item['claim'],
            'result': item.get('result'),
            'error': item.get('error')
        }
        results_data['results'].append(result_entry)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not save results to JSON: {str(e)}{Colors.RESET}")
        return None

async def main_async():
    """Async main function for parallel fact-checking"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'🤖 AI-POWERED FACT CHECKER'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 70}{Colors.RESET}\n")
    
    # Load cache
    load_cache()
    
    # Step 1: Record audio
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'[STEP 1/4]'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'🎤 Recording Audio'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}\n")
    try:
        audio_file = record_audio(duration=10, filename="recording.wav")
        
        if not audio_file or not os.path.exists(audio_file):
            print(f"{Colors.BRIGHT_RED}✗ [ERROR]{Colors.RESET} Failed to record audio. Cannot continue.")
            return
        
        print(f"{Colors.BRIGHT_GREEN}✓ [SUCCESS]{Colors.RESET} Audio recorded: {Colors.BRIGHT_CYAN}{audio_file}{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}✗ [ERROR]{Colors.RESET} Audio recording failed: {Colors.RED}{str(e)}{Colors.RESET}")
        return
    
    # Step 2: Transcribe audio
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'[STEP 2/4]'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'📝 Transcribing Audio'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}\n")
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print(f"\n{Colors.BRIGHT_RED}✗ [ERROR]{Colors.RESET} ANTHROPIC_API_KEY not found in .env file")
        print(f"  {Colors.YELLOW}Please add your API key to the .env file.{Colors.RESET}")
        return
    
    try:
        # Initialize Anthropic client
        client = Anthropic(api_key=api_key)
        
        # Transcribe the audio
        transcribed_text = transcribe_audio(audio_file, client)
        
        if not transcribed_text:
            print(f"{Colors.BRIGHT_RED}✗ [ERROR]{Colors.RESET} Transcription failed. Cannot continue.")
            return
        
        print(f"{Colors.BRIGHT_GREEN}✓ [SUCCESS]{Colors.RESET} Transcription completed!")
        
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}✗ [ERROR]{Colors.RESET} Transcription failed: {Colors.RED}{str(e)}{Colors.RESET}")
        return
    
    # Step 3: Extract claims from transcript
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'[STEP 3/4]'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'🔍 Extracting Claims'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}\n")
    
    try:
        claims = extract_claims(transcribed_text, client)
        
        if not claims:
            print(f"\n{Colors.YELLOW}[INFO] No factual claims found in the transcript.{Colors.RESET}")
            print_boxed_text("📝 ORIGINAL TRANSCRIPT", transcribed_text, width=70)
            return
        
        print(f"{Colors.BRIGHT_GREEN}✓ [SUCCESS]{Colors.RESET} Extracted {Colors.BOLD}{Colors.BRIGHT_YELLOW}{len(claims)}{Colors.RESET} claim(s)")
        
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}✗ [ERROR]{Colors.RESET} Claim extraction failed: {Colors.RED}{str(e)}{Colors.RESET}")
        return
    
    # Step 4: Fact-check claims in parallel
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'[STEP 4/4]'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'✅ Fact-Checking Claims (Parallel)'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{'═' * 70}{Colors.RESET}\n")
    
    # Initialize progress tracker
    progress = ProgressTracker(len(claims))
    
    # Progress callback
    def progress_callback(claim, status):
        progress.update(claim, status)
    
    # Create async tasks for parallel processing
    tasks = [
        fact_check_claim_async(claim, client, use_cache=True, progress_callback=progress_callback)
        for claim in claims
    ]
    
    # Execute all tasks in parallel
    print(f"{Colors.BRIGHT_CYAN}Starting parallel fact-checking of {len(claims)} claim(s)...{Colors.RESET}\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    print("\n")  # New line after progress bar
    fact_check_results = []
    
    for i, (claim, result) in enumerate(zip(claims, results), 1):
        if isinstance(result, Exception):
            print(f"{Colors.BRIGHT_RED}[{i}/{len(claims)}] Error:{Colors.RESET} {str(result)}")
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
    
    # Save results to JSON
    json_filename = save_results_to_json(transcribed_text, claims, fact_check_results)
    if json_filename:
        print(f"\n{Colors.BRIGHT_GREEN}✓{Colors.RESET} Results saved to: {Colors.BRIGHT_CYAN}{json_filename}{Colors.RESET}")
    
    # Print final results (keep existing formatting)
    print_results(transcribed_text, claims, fact_check_results)

def print_results(transcribed_text, claims, fact_check_results):
    """Print the final results in a formatted way"""
    # Print final results with enhanced formatting
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'FINAL RESULTS'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 70}{Colors.RESET}\n")
    
    # Print transcript in a box
    print_boxed_text("📝 ORIGINAL TRANSCRIPT", transcribed_text, width=70)
    
    if claims:
        # Print claims with highlighting
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}{'─' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}📋 CLAIMS EXTRACTED ({len(claims)}){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}{'─' * 70}{Colors.RESET}\n")
        
        for i, claim in enumerate(claims, 1):
            print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}[{i}]{Colors.RESET} {Colors.BOLD}{Colors.WHITE}{claim}{Colors.RESET}\n")
        
        # Print fact-check results with color coding
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}{'─' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}🔍 FACT-CHECK RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}{'─' * 70}{Colors.RESET}\n")
        
        for i, item in enumerate(fact_check_results, 1):
            claim = item['claim']
            result = item.get('result')
            error = item.get('error')
            
            print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}[{i}]{Colors.RESET} {Colors.BOLD}{Colors.WHITE}Claim:{Colors.RESET} \"{claim}\"")
            print()
            
            if result:
                verdict = result['verdict'].upper()
                confidence = result['confidence']
                explanation = result['explanation']
                sources = result.get('sources', [])
                
                # Get color and icon based on verdict
                verdict_color = get_verdict_color(verdict, confidence)
                verdict_icon = get_verdict_icon(verdict)
                
                # Print verdict with color
                print(f"  {Colors.BOLD}{verdict_color}{verdict_icon} Verdict:{Colors.RESET} {Colors.BOLD}{verdict_color}{verdict}{Colors.RESET} ({Colors.BOLD}{confidence}%{Colors.RESET} confidence)")
                
                # Print explanation
                print(f"  {Colors.BOLD}{Colors.CYAN}💡 Explanation:{Colors.RESET} {Colors.WHITE}{explanation}{Colors.RESET}")
                
                # Print sources as URLs
                if sources:
                    print(f"  {Colors.BOLD}{Colors.CYAN}🔗 Sources:{Colors.RESET}")
                    for source in sources[:5]:  # Show first 5 sources
                        # Check if it looks like a URL or domain
                        if any(char in source for char in ['.com', '.org', '.net', '.edu', 'http', 'www']):
                            print(f"     {Colors.BRIGHT_CYAN}→{Colors.RESET} {format_url(source)}")
                        else:
                            print(f"     {Colors.BRIGHT_CYAN}→{Colors.RESET} {Colors.WHITE}{source}{Colors.RESET}")
            else:
                print(f"  {Colors.BRIGHT_RED}❌ [ERROR]{Colors.RESET} {Colors.RED}{error if error else 'Fact-check failed'}{Colors.RESET}")
            
            print()
            if i < len(fact_check_results):
                print(f"{Colors.BLACK}{Colors.BG_BLACK}{'─' * 70}{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}{'─' * 70}{Colors.RESET}\n")
    
    # Final completion message
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'✓ Pipeline completed successfully!'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}{'═' * 70}{Colors.RESET}\n")

def main():
    """Main function that runs the async pipeline"""
    asyncio.run(main_async())

if __name__ == "__main__":
    # Enable Windows event loop policy if needed
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()


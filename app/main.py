import logging
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv
import tempfile
import shutil
import re
from time import sleep
import json
import time
import uuid
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import Dict, List, Optional
import sys

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get port from environment variable
try:
    PORT = int(os.getenv("PORT", "8000"))
    logger.info(f"PORT environment variable resolved to: {PORT}")
except ValueError as e:
    logger.error(f"Invalid PORT value: {os.getenv('PORT')}")
    PORT = 8000
    logger.info(f"Defaulting to port {PORT}")

api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"Environment variables loaded. API Key present: {bool(api_key)}")

if not api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please check your .env file.")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Recipe Backend",
    description="API for processing YouTube cooking videos into recipes",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting application on port {PORT}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Test if we can write to the data directory
    try:
        test_file = os.path.join(DATA_DIR, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"Successfully verified write access to {DATA_DIR}")
    except Exception as e:
        logger.error(f"Failed to write to data directory: {str(e)}")
        
    # Test OpenAI connection
    try:
        logger.info("Testing OpenAI API connection...")
        client.models.list()
        logger.info("OpenAI API connection successful")
    except Exception as e:
        logger.error(f"OpenAI API connection failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    try:
        return {
            "status": "running",
            "port": PORT,
            "environment": {
                "python_version": sys.version,
                "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
                "data_dir": DATA_DIR,
                "recipes_count": len(recipes),
                "cwd": os.getcwd(),
                "port_env": os.getenv("PORT", "not set")
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)  # Initialize with explicit configuration
logger.info("OpenAI client initialized")

# In-memory storage for recipes
recipes: Dict[str, dict] = {}

# Define data directory
DATA_DIR = os.getenv('DATA_DIR', '/tmp')
RECIPES_FILE = os.path.join(DATA_DIR, 'recipes.json')

def save_recipe(recipe: dict):
    """Save recipe to storage."""
    recipes[recipe['id']] = recipe
    save_recipes_to_file()

def load_recipes_from_file() -> Dict[str, dict]:
    """Load recipes from file storage."""
    try:
        with open(RECIPES_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_recipes_to_file():
    """Save recipes to file storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RECIPES_FILE, 'w') as f:
        json.dump(recipes, f)

# Load existing recipes on startup
recipes = load_recipes_from_file()
logger.info(f"Loaded {len(recipes)} recipes from storage")

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 5  # Total number of main steps in the pipeline
        
    def update_progress(self, step_name: str, status: str = "started"):
        self.current_step += 1
        elapsed = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100
        logger.info(f"[{progress:.1f}%] Step {self.current_step}/{self.total_steps} - {step_name} {status} (Elapsed: {elapsed:.2f}s)")

def clean_youtube_url(url: str) -> str:
    """Clean and validate YouTube URL."""
    # Remove any whitespace
    url = url.strip()
    
    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and short URLs
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',  # Shorts URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return f'https://youtube.com/watch?v={video_id}'
    
    raise ValueError("Invalid YouTube URL format")

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and short URLs
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',  # Shorts URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def get_video_info(url: str) -> dict:
    """Get video metadata using yt-dlp without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', ''),
                'thumbnail': info.get('thumbnail', ''),
                'duration': info.get('duration', 0),
            }
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise ValueError(f"Failed to get video info: {str(e)}")

def get_transcript(video_id: str) -> dict:
    """Get transcript using youtube-transcript-api."""
    try:
        # Try to get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # First try to get manual English transcript
            transcript = transcript_list.find_manually_created_transcript(['en'])
            logger.info("✓ Found manual English transcript")
        except NoTranscriptFound:
            try:
                # Then try auto-generated English transcript
                transcript = transcript_list.find_generated_transcript(['en'])
                logger.info("✓ Found auto-generated English transcript")
            except NoTranscriptFound:
                try:
                    # Finally, try to get any transcript and translate it to English
                    transcript = transcript_list.find_transcript(['en'])
                    if transcript.language_code != 'en':
                        logger.info(f"Found transcript in {transcript.language_code}, translating to English")
                        transcript = transcript.translate('en')
                except NoTranscriptFound:
                    logger.info("✗ No transcripts found")
                    return None
        
        # Get the actual transcript data
        transcript_data = transcript.fetch()
        
        # Combine all text entries
        full_text = ' '.join(entry['text'] for entry in transcript_data)
        
        return {
            'transcript': full_text,
            'source': 'youtube_transcript_api'
        }
        
    except TranscriptsDisabled:
        logger.info("✗ Transcripts are disabled for this video")
        return None
    except Exception as e:
        logger.warning(f"Error getting transcript: {str(e)}")
        return None

def download_audio_for_whisper(url: str, output_path: str, progress: ProgressTracker) -> str:
    """Download audio file for Whisper transcription."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [lambda d: logger.info(f"Download progress: {d.get('_percent_str', 'N/A')} of {d.get('_total_bytes_str', 'N/A')}")],
    }
    
    progress.update_progress("Audio download", "started")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            if 'requested_downloads' in info:
                downloaded_file = info['requested_downloads'][0]['filepath']
            else:
                filename = ydl.prepare_filename(info)
                base, _ = os.path.splitext(filename)
                downloaded_file = f"{base}.mp3"
                
            progress.update_progress("Audio download", "completed")
            return downloaded_file
            
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        raise ValueError(f"Failed to download audio: {str(e)}")

def generate_unique_id(video_url: str) -> str:
    """Generate a unique ID for a recipe using video URL and timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    unique_component = str(uuid.uuid5(uuid.NAMESPACE_URL, video_url))[:8]
    return f"{unique_component}-{timestamp}"

@app.get("/recipes")
async def get_recipes() -> List[dict]:
    """Get all recipes."""
    return list(recipes.values())

@app.get("/recipes/{recipe_id}")
async def get_recipe(recipe_id: str) -> dict:
    """Get a specific recipe by ID."""
    if recipe_id not in recipes:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipes[recipe_id]

@app.delete("/recipes/{recipe_id}")
async def delete_recipe(recipe_id: str):
    """Delete a recipe by ID."""
    if recipe_id not in recipes:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    try:
        # Remove from memory
        del recipes[recipe_id]
        # Save to file
        save_recipes_to_file()
        logger.info(f"Recipe {recipe_id} deleted successfully")
        return {"message": "Recipe deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting recipe {recipe_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete recipe")

@app.post("/process-video")
async def process_video(video_url: str = Form(...)):
    try:
        progress = ProgressTracker()
        logger.info(f"Starting video processing pipeline for URL: {video_url}")
        progress.update_progress("Initialization", "completed")
        
        # Clean and validate the URL
        clean_url = clean_youtube_url(video_url)
        video_id = extract_video_id(clean_url)
        logger.info(f"Extracted video ID: {video_id}")

        # Get video metadata
        video_info = get_video_info(clean_url)
        logger.info(f"Got video info: {video_info['title']}")

        # Create temporary directory for audio file if needed
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Try to get transcript using youtube-transcript-api
            transcript_info = get_transcript(video_id)
            
            if transcript_info:
                logger.info(f"Using {transcript_info['source']} for transcription")
                transcription_text = transcript_info['transcript']
                progress.update_progress("Transcription", "completed")
            else:
                # Fall back to Whisper if no transcript available
                logger.info("No YouTube transcripts available. Falling back to Whisper")
                # Download audio for Whisper
                audio_file_path = download_audio_for_whisper(clean_url, temp_dir, progress)
                logger.info(f"Downloaded audio: {audio_file_path}")
                
                # Transcribe with Whisper
                progress.update_progress("Transcription", "started")
                logger.info("Starting audio transcription with Whisper")
                with open(audio_file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                transcription_text = transcription.text
                logger.info("Whisper transcription completed successfully")
                progress.update_progress("Transcription", "completed")

            # Generate recipe from transcription
            progress.update_progress("Recipe generation", "started")
            logger.info("Starting recipe generation from transcription")
            recipe_prompt = f"""Create a recipe from the following cooking video transcription. The transcription might be in a different language, but always create the recipe in English.

Format your response STRICTLY as a JSON object with this exact structure:
{{
    "title": "Recipe Title",
    "description": "A brief description of the dish",
    "ingredients": ["ingredient 1", "ingredient 2", ...],
    "instructions": ["step 1", "step 2", ...],
    "tips": ["tip 1", "tip 2", ...]
}}

Transcription:
{transcription_text}"""
            
            recipe_response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a professional chef who creates detailed recipes from video transcriptions. ALWAYS format your response as a valid JSON object. If the transcription is not in English, first understand it, then create the recipe in English. Double-check that your response is valid JSON before returning it."},
                    {"role": "user", "content": recipe_prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            recipe_content = recipe_response.choices[0].message.content
            logger.info("Raw GPT response received")
            logger.debug(f"Raw recipe content: {recipe_content}")
            
            # Validate JSON before proceeding
            recipe_data = json.loads(recipe_content)
            required_fields = ["title", "description", "ingredients", "instructions"]
            missing_fields = [field for field in required_fields if field not in recipe_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in recipe: {', '.join(missing_fields)}")
            
            logger.info("Recipe generation completed successfully")
            progress.update_progress("Recipe generation", "completed")
            
            # Parse the recipe response and format it according to frontend needs
            progress.update_progress("Final processing", "started")
            result = {
                "id": generate_unique_id(video_url),
                "title": recipe_data["title"],
                "description": recipe_data["description"],
                "youtubeUrl": clean_url,
                "ingredients": recipe_data["ingredients"],
                "instructions": recipe_data["instructions"],
                "tips": recipe_data.get("tips", []),
                "video_title": video_info['title'],
                "thumbnail": video_info['thumbnail'],
                "created_at": datetime.now().isoformat()
            }
            
            # Save the recipe
            save_recipe(result)
            
            progress.update_progress("Final processing", "completed")
            logger.info(f"Processing completed successfully in {time.time() - progress.start_time:.2f} seconds")
            return result

    except ValueError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse recipe data")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.getenv("PORT", "8000"))
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=port,
            log_level="debug",
            timeout_keep_alive=75
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 
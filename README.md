# Recipe PWA Backend

This is the backend service for the Recipe PWA application. It uses FastAPI to provide endpoints for processing YouTube videos and generating recipes.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Server

To start the server, run:
```bash
python -m app.main
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /process-video
Process a YouTube video URL to generate a recipe.

Request body:
```json
{
    "video_url": "https://www.youtube.com/watch?v=..."
}
```

Response:
```json
{
    "transcription": "Transcribed text from the video...",
    "recipe": "Generated recipe from the transcription..."
}
``` 
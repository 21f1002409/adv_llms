from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from google.genai import types
import base64
import os
import requests
import json
from typing import List, Optional, Union, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Unified AI API with Function Calling", description="Single endpoint with intelligent tool routing")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not GOOGLE_API_KEY or not SARVAM_API_KEY:
    raise ValueError("API keys not set")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Unified request model
class UnifiedRequest(BaseModel):
    input_text: str
    input_image_base64: Optional[str] = None
    input_audio_base64: Optional[str] = None
    context: Optional[str] = None  # Additional context for the request
    language_code: Optional[str] = "en-IN"  # For TTS
    conversation_history: Optional[List[dict]] = []

class UnifiedResponse(BaseModel):
    action_taken: str
    result: Any
    function_called: str
    success: bool
    message: str

# Function definitions for the LLM
AVAILABLE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Analyze sentiment of text to determine if it's positive, negative, or neutral",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze for sentiment"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_image",
            "description": "Extract text from an image or describe image content",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image to transcribe or describe"
                    }
                },
                "required": ["image_base64"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Convert text to speech audio in specified language",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech"
                    },
                    "language_code": {
                        "type": "string",
                        "description": "Language code (e.g., en-IN, hi-IN, te-IN)",
                        "default": "en-IN"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Voice speaker name",
                        "default": "Meera"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_embeddings",
            "description": "Generate vector embeddings for text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to generate embeddings for"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chat_conversation",
            "description": "Have a conversational chat with the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "User's message for conversation"
                    },
                    "conversation_history": {
                        "type": "array",
                        "description": "Previous conversation history",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["message"]
            }
        }
    }
]

# Individual function implementations
async def analyze_sentiment(text: str):
    """Analyze sentiment of text"""
    try:
        prompt = f"""
        Analyze the sentiment of the following text and respond with only "positive", "negative", or "neutral":
        Text: "{text}"
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        result_text = response.text.lower().strip()
        sentiment = "neutral"
        
        if "positive" in result_text:
            sentiment = "positive"
        elif "negative" in result_text:
            sentiment = "negative"
            
        return {"sentiment": sentiment, "confidence": result_text}
        
    except Exception as e:
        raise Exception(f"Error analyzing sentiment: {str(e)}")

async def transcribe_image(image_base64: str):
    """Transcribe or describe image content"""
    try:
        image_data = base64.b64decode(image_base64)
        
        image_part = types.Part.from_bytes(
            data=image_data, 
            mime_type="image/jpeg"
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Please transcribe any text you see in this image. If there's no text, describe what you see in the image in detail.",
                image_part
            ]
        )
        
        return {"transcription": response.text}
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

async def text_to_speech(text: str, language_code: str = "en-IN", speaker: str = "Meera"):
    """Convert text to speech"""
    try:
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "target_language_code": language_code,
            "speaker": speaker
        }
        
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"TTS API error: {response.text}")
        
        result = response.json()
        
        return {
            "audio_base64": result["audios"][0],
            "language": language_code,
            "speaker": speaker
        }
        
    except Exception as e:
        raise Exception(f"Error generating speech: {str(e)}")

async def generate_embeddings(text: str):
    """Generate embeddings for text"""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text
        )
        
        embeddings = result.embeddings[0].values if result.embeddings else []
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings)
        }
        
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

async def chat_conversation(message: str, conversation_history: List[dict] = None):
    """Handle chat conversation"""
    try:
        if conversation_history is None:
            conversation_history = []
            
        conversation_context = ""
        
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                conversation_context += f"User: {content}\n"
            elif role == "assistant":
                conversation_context += f"Assistant: {content}\n"
        
        conversation_context += f"User: {message}\nAssistant: "
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation_context
        )
        
        assistant_response = response.text.strip()
        
        updated_history = conversation_history.copy()
        updated_history.append({"role": "user", "content": message})
        updated_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "response": assistant_response,
            "conversation_history": updated_history
        }
        
    except Exception as e:
        raise Exception(f"Error in chat: {str(e)}")

# Function dispatcher
FUNCTION_DISPATCHER = {
    "analyze_sentiment": analyze_sentiment,
    "transcribe_image": transcribe_image,
    "text_to_speech": text_to_speech,
    "generate_embeddings": generate_embeddings,
    "chat_conversation": chat_conversation
}

# Main unified endpoint
@app.post("/api", response_model=UnifiedResponse)
async def unified_api(request: UnifiedRequest):
    """
    Unified API endpoint that uses LLM to determine which function to call
    based on user input and context.
    """
    try:
        # Prepare the prompt for function calling
        user_prompt = f"""
        User input: {request.input_text}
        
        Additional context: {request.context if request.context else 'None'}
        
        Available data:
        - Text input: {request.input_text}
        - Image provided: {'Yes' if request.input_image_base64 else 'No'}
        - Audio provided: {'Yes' if request.input_audio_base64 else 'No'}
        - Language preference: {request.language_code}
        
        Based on the user's request, determine which function to call to best help them.
        """
        
        # Call Gemini with function calling
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            tools=AVAILABLE_FUNCTIONS
        )
        
        # Check if the model wants to call a function
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            # Add missing parameters from request
            if function_name == "transcribe_image" and "image_base64" not in function_args:
                if request.input_image_base64:
                    function_args["image_base64"] = request.input_image_base64
                else:
                    raise HTTPException(status_code=400, detail="Image required for transcription")
            
            if function_name == "text_to_speech":
                function_args["language_code"] = request.language_code
                
            if function_name == "chat_conversation":
                function_args["conversation_history"] = request.conversation_history
            
            # Execute the function
            if function_name in FUNCTION_DISPATCHER:
                result = await FUNCTION_DISPATCHER[function_name](**function_args)
                
                return UnifiedResponse(
                    action_taken=f"Called {function_name}",
                    result=result,
                    function_called=function_name,
                    success=True,
                    message=f"Successfully executed {function_name}"
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown function: {function_name}")
        
        else:
            # No function call needed, return direct response
            return UnifiedResponse(
                action_taken="Direct response",
                result={"response": response.text},
                function_called="none",
                success=True,
                message="Provided direct response without function call"
            )
            
    except Exception as e:
        return UnifiedResponse(
            action_taken="Error occurred",
            result={"error": str(e)},
            function_called="none",
            success=False,
            message=f"Error processing request: {str(e)}"
        )

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Unified AI API"}

# Available functions info
@app.get("/functions")
async def get_available_functions():
    """Get list of available functions"""
    return {
        "available_functions": [func["function"]["name"] for func in AVAILABLE_FUNCTIONS],
        "function_details": AVAILABLE_FUNCTIONS
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Unified AI API with Function Calling",
        "description": "Single endpoint that intelligently routes to appropriate AI functions",
        "main_endpoint": "/api",
        "supported_inputs": ["text", "image (base64)", "audio (base64)"],
        "available_functions": [func["function"]["name"] for func in AVAILABLE_FUNCTIONS]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

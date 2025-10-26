"""
Google Gemini API Integration - PRODUCTION READY FIX
Safety settings corrected from BLOCK_NONE to BLOCK_MEDIUM_AND_ABOVE
"""
from typing import Optional
import google.generativeai as genai
import os
import logging

class GeminiLLM:
    """Wrapper for Google's Gemini API with correct safety handling."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, **kwargs):
        """Initialize the Gemini LLM wrapper."""
        self.logger = logging.getLogger(__name__)
        
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GEMINI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.kwargs = {
            'temperature': kwargs.get('temperature', 0.2),
            'max_output_tokens': kwargs.get('max_output_tokens', 2048),
        }
        
        try:
            genai.configure(api_key=self.api_key)
            self.logger.info(f"Configured Gemini API with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise
        
        self._setup_model()
    
    def _setup_model(self):
        """Set up the Gemini model with CORRECTED safety settings."""
        try:
            self.logger.info(f"Initializing Gemini model: {self.model_name}")
            
            generation_config = {
                'temperature': self.kwargs['temperature'],
                'max_output_tokens': self.kwargs['max_output_tokens'],
                'top_p': 0.95,
                'top_k': 40
            }
            
            # ===== CRITICAL FIX: CHANGED FROM BLOCK_NONE TO BLOCK_MEDIUM_AND_ABOVE =====
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  # ← FIXED: Was BLOCK_NONE
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  # ← FIXED: Was BLOCK_NONE
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  # ← FIXED: Was BLOCK_NONE
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  # ← FIXED: Was BLOCK_NONE
                },
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info("✅ Model initialized with BLOCK_MEDIUM_AND_ABOVE safety settings")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise RuntimeError(f"Gemini model initialization failed: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt with comprehensive error handling."""
        from google.api_core import exceptions as google_exceptions
        
        try:
            self.logger.debug(f"Generating response (prompt length: {len(prompt)} chars)")
            
            generation_config = {
                'temperature': float(kwargs.get('temperature', self.kwargs['temperature'])),
                'max_output_tokens': int(kwargs.get('max_output_tokens', self.kwargs['max_output_tokens'])),
                'top_p': 0.95,
                'top_k': 40
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=False
            )
            
            # ===== CHECK FINISH_REASON BEFORE ACCESSING TEXT =====
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    self.logger.debug(f"Finish reason code: {finish_reason}")
                    
                    if finish_reason == 2:  # SAFETY
                        safety_ratings = getattr(candidate, 'safety_ratings', [])
                        self.logger.warning(f"Response blocked (SAFETY): {safety_ratings}")
                        return "The response was blocked by content safety filters."
                    
                    elif finish_reason == 3:  # RECITATION
                        self.logger.warning("Response blocked (RECITATION)")
                        return "Response blocked: Detected copyrighted content."
                    
                    elif finish_reason == 4:  # OTHER
                        self.logger.warning("Response blocked (OTHER)")
                        return "Response could not be generated. Please try again."
            
            # EXTRACT TEXT SAFELY
            try:
                if hasattr(response, 'text') and response.text:
                    self.logger.debug("✅ Successfully extracted text from response")
                    return response.text
                
                if hasattr(response, 'parts') and response.parts:
                    parts = []
                    for part in response.parts:
                        if hasattr(part, 'text') and part.text:
                            parts.append(part.text)
                    if parts:
                        self.logger.debug("✅ Successfully extracted text from parts")
                        return '\n'.join(parts)
                
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                parts = []
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        parts.append(part.text)
                                if parts:
                                    self.logger.debug("✅ Successfully extracted text from candidates")
                                    return '\n'.join(parts)
                
                self.logger.error("No valid text found in response")
                return "No response generated - model returned empty result."
            
            except AttributeError as e:
                self.logger.error(f"Error parsing response: {str(e)}")
                if hasattr(response, 'prompt_feedback'):
                    return f"Prompt blocked: {response.prompt_feedback}"
                return "Error: Could not extract text from response."
        
        except google_exceptions.InvalidArgument as e:
            self.logger.error(f"Invalid argument: {str(e)}")
            if "blocked" in str(e).lower():
                return "Request blocked by safety filters."
            return f"Invalid request: {str(e)}"
        
        except google_exceptions.ResourceExhausted as e:
            self.logger.error(f"Quota exceeded: {str(e)}")
            return "API quota exceeded. Check Google AI Studio account."
        
        except google_exceptions.PermissionDenied as e:
            self.logger.error(f"Permission denied: {str(e)}")
            return "API key invalid or has insufficient permissions."
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

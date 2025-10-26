"""
Hugging Face LLM integration for the Code Explainer Agent.
"""
import os
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class HuggingFaceLLM:
    """Wrapper for local Hugging Face models."""
    
    def __init__(self, model_name: str = "gpt2", **kwargs):
        """Initialize the local Hugging Face model.
        
        Args:
            model_name: Name of the local model to use (default: "gpt2")
            **kwargs: Additional arguments to pass to the model
        """
        self.model_name = model_name
        self.kwargs = {
            'temperature': 0.2,
            'max_length': 100,
            **kwargs  # Override defaults with any provided kwargs
        }
        
        # Set up the model and tokenizer
        self._setup_model()
    
    def _setup_model(self):
        """Set up the model and tokenizer."""
        try:
            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **{k: v for k, v in self.kwargs.items() if k not in ['temperature', 'max_length']}
            )
            
            # Move to GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda':
                self.model = self.model.to(self.device)
                
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            The generated text
        """
        try:
            # Get generation parameters
            temperature = kwargs.get('temperature', self.kwargs['temperature'])
            max_length = kwargs.get('max_length', self.kwargs['max_length'])
            
            # Encode the input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            if self.device == 'cuda':
                input_ids = input_ids.to(self.device)
            
            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=max(0.1, min(1.0, float(temperature))),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_length']}
                )
            
            # Decode and return the output
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating text: {str(e)}\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)

# Example usage
if __name__ == "__main__":
    # Initialize with a free model (no API key needed for public models)
    llm = HuggingFaceLLM(model_name="gpt2")  # Using GPT-2 as an example
    
    # Generate text
    response = llm.generate("Explain this code:\ndef hello_world():\n    print('Hello, World!')")
    print("Generated response:", response)

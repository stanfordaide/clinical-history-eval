import re

class OutputParser:
    """Robust parser for extracting structured information from model outputs."""
    
    def __init__(self):
        # More flexible regex patterns for each field
        self.patterns = {
            'pmh': r'(?:past medical history|pmh):\s*(.*?)(?:\d+\.|$)',
            'what': r'(?:what):\s*(.*?)(?:\d+\.|$)',
            'when': r'(?:when):\s*(.*?)(?:\d+\.|$)',
            'where': r'(?:where):\s*(.*?)(?:\d+\.|$)',
            'concern': r'(?:clinical concern|concern):\s*(.*?)(?:\d+\.|$)'
        }
        
        # Define phrases that indicate absence/non-applicability
        self.null_phrases = {
            'not included',
        }
        
    def parse(self, text: str) -> dict:
        """Parse the model output and extract fields with improved robustness."""
        # Original parsing logic remains the same
        text = text.lower()
        text = text.split("### task")[-1]
        text = text.split("###---end---###")[0]
        
        result = {
            'input': 'not included',
            'pmh': 'not included',
            'what': 'not included',
            'when': 'not included',
            'where': 'not included',
            'concern': 'not included'
        }
        
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if value and value != "not included":
                    result[field] = value
        
        return result
    
    def parse_binary(self, text: str) -> dict:
        """Parse the model output into binary values (0 for absent/NA, 1 for present)."""
        parsed = self.parse(text)
        binary_result = {}
        
        for field, value in parsed.items():
            if field == 'input':
                continue
            # Convert to binary: 0 if exactly matches null phrases, 1 otherwise
            value_lower = value.lower().strip()
            binary_result[field] = 0 if value_lower in self.null_phrases else 1
            
        return binary_result
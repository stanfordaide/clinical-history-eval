import re

# class OutputParser:
#     """Simple parser for extracting structured information from model outputs."""
    
#     def __init__(self):
#         self.regex_pattern = r'note\s\d+:\n([\s\S]*?)response\s\d+:\n\d+.past medical history:([\s\S]*?)\d+.what:([\s\S]*?)\d+.when:([\s\S]*?)\d+.where:([\s\S]*?)\d+.clinical concern:(.*)'
        
#     def parse(self, text: str) -> dict:
#         """Parse the model output and extract fields."""
#         # Split at ### Task and take the last part
#         text = text.split("### Task")[-1].lower()
        
#         # Extract fields using regex
#         match = re.findall(self.regex_pattern, text)
        
#         if match and len(match[0]) == 6:  # Should have 6 groups
#             return {
#                 'input': match[0][0].strip(),
#                 'pmh': match[0][1].strip(),
#                 'what': match[0][2].strip(),
#                 'when': match[0][3].strip(),
#                 'where': match[0][4].strip(),
#                 'concern': match[0][5].strip()
#             }
        
#         return {
#             'input': 'not included',
#             'pmh': 'not included',
#             'what': 'not included',
#             'when': 'not included',
#             'where': 'not included',
#             'concern': 'not included'
#         }

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
        
    def parse(self, text: str) -> dict:
        """Parse the model output and extract fields with improved robustness."""
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Split at ### Task and take the last part
        text = text.split("### task")[-1]
        text = text.split("###---end---###")[0]
        
        # Initialize result dictionary
        result = {
            'input': 'not included',
            'pmh': 'not included',
            'what': 'not included',
            'when': 'not included',
            'where': 'not included',
            'concern': 'not included'
        }
        
        # Extract each field using individual patterns
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if value and value != "not included":
                    result[field] = value
        
        return result
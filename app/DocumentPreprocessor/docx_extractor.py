from .text_extractor import TextExtractor
from docx import Document
import io
import re

class DocxExtractor(TextExtractor):
    def extract_text(self):
        try:
            # Read the file content
            file_content = self.file.file.read()
            # Reset file pointer for potential future reads
            self.file.file.seek(0)
            
            # Create a BytesIO object from the file content
            file_stream = io.BytesIO(file_content)
            doc = Document(file_stream)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return self._final_cleanup(text)
        except Exception as e:
            print(f"Error processing DOCX {self.file.filename}: {e}")
            return ""
        
    def _final_cleanup(self, text: str) -> str:
        # Remove excessive consecutive newlines but preserve paragraph structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        
        # Don't remove lines with content - just clean up spacing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip():
                cleaned_lines.append(line.strip())
            else:
                cleaned_lines.append('') 
        
        return '\n'.join(cleaned_lines)
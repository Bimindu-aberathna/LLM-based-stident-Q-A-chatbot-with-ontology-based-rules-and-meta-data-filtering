from .text_extractor import TextExtractor
import fitz  # PyMuPDF
import re

class PDFExtractor(TextExtractor):
    def extract_text(self) -> str:
        try:
            # Read the file content
            file_content = self.file.file.read()
            # Reset file pointer for potential future reads
            self.file.file.seek(0)
            
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page_num, page in enumerate(doc):
                # Get page dimensions
                page_rect = page.rect
                page_height = page_rect.height
                page_width = page_rect.width
                
                content_rect = fitz.Rect(
                    0, 
                    page_height * 0.07,  # 7% from top
                    page_width, 
                    page_height * 0.95 # 5% to bottom
                )
                
                # Extract text only from content area
                page_text = page.get_text(clip=content_rect)
                
                text += page_text + "\n"
            
            doc.close()
            return self._final_cleanup(text)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
        
    def _final_cleanup(self, text: str) -> str:
        # Remove excessive consecutive newlines but preserve paragraph structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        
        # Don't remove lines with content - just clean up spacing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Keep all non-empty lines, just clean whitespace
            if line.strip():
                cleaned_lines.append(line.strip())
            else:
                cleaned_lines.append('')  # Preserve empty lines for structure
        
        return '\n'.join(cleaned_lines)    
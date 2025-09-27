from abc import ABC, abstractmethod


class TextExtractor(ABC):
    file = None

    def __init__(self, file):
        self.supported_formats = ['.pdf', '.docx', '.txt']
        self.file = file
        self.validate_file_format()

    def validate_file_format(self):
        if not any(self.file.filename.endswith(ext) for ext in self.supported_formats):
            raise ValueError(f"Unsupported file format. Supported formats are: {', '.join(self.supported_formats)}")
        return True
    
    @abstractmethod
    def extract_text(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    
    

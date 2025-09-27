import PyPDF2
from io import BytesIO

def process_pdf(file_bytes: bytes) -> str:
    try:
        pdf_file = BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        return text_content.strip()
    except Exception as e:
        
        return file_bytes.decode("utf-8", errors="ignore")

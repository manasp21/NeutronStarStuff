#!/usr/bin/env python3
"""
Simple PDF text extraction using built-in tools.
"""

import subprocess
import os
import sys

def extract_pdf_with_strings(pdf_path):
    """Extract text from PDF using the 'strings' command."""
    try:
        result = subprocess.run(['strings', pdf_path], 
                              capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"strings command failed: {result.stderr}")
            return None
    except FileNotFoundError:
        print("'strings' command not found")
        return None
    except Exception as e:
        print(f"Error using strings: {e}")
        return None

def extract_pdf_with_pdftotext(pdf_path):
    """Extract text from PDF using pdftotext if available."""
    try:
        result = subprocess.run(['pdftotext', pdf_path, '-'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"pdftotext failed: {result.stderr}")
            return None
    except FileNotFoundError:
        print("pdftotext not found")
        return None
    except Exception as e:
        print(f"Error using pdftotext: {e}")
        return None

def try_python_pdf_libraries(pdf_path):
    """Try to use Python PDF libraries if available."""
    text_content = ""
    
    # Try PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
        return text_content
    except ImportError:
        pass
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
    
    # Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_content += page.get_text() + "\n"
        doc.close()
        return text_content
    except ImportError:
        pass
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
    
    return None

def extract_pdf_text(pdf_path):
    """Extract text from PDF using available methods."""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None
    
    print(f"Attempting to extract text from: {pdf_path}")
    
    # Try Python libraries first
    text = try_python_pdf_libraries(pdf_path)
    if text and text.strip():
        print("Successfully extracted text using Python libraries")
        return text
    
    # Try pdftotext
    text = extract_pdf_with_pdftotext(pdf_path)
    if text and text.strip():
        print("Successfully extracted text using pdftotext")
        return text
    
    # Try strings as last resort
    text = extract_pdf_with_strings(pdf_path)
    if text and text.strip():
        print("Extracted text using strings command (may be incomplete)")
        return text
    
    print("Failed to extract text from PDF")
    return None

def main():
    """Main function to test PDF extraction."""
    pdf_path = "EoS_step_by_step.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    text = extract_pdf_text(pdf_path)
    
    if text:
        # Save extracted text
        output_file = pdf_path.replace('.pdf', '_extracted.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text saved to: {output_file}")
        
        # Show first 1000 characters
        print("\nFirst 1000 characters of extracted text:")
        print("=" * 50)
        print(text[:1000])
        print("=" * 50)
    else:
        print("Could not extract text from PDF")

if __name__ == "__main__":
    main()
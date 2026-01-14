i# scripts/sanity_check.py
import sys

def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)

    # Core imports you said you need
    import pandas as pd
    import numpy as np
    import fitz  # PyMuPDF
    from PIL import Image
    import pypdf
    import docx
    import openpyxl
    import tqdm
    import pytesseract
    import pdf2image

    print("Imports OK.")
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
    print("PyMuPDF:", fitz.__doc__.splitlines()[0] if fitz.__doc__ else "ok")
    print("Pillow:", Image.__version__)
    print("pypdf:", getattr(pypdf, "__version__", "unknown"))
    print("python-docx:", getattr(docx, "__version__", "unknown"))
    print("openpyxl:", openpyxl.__version__)

if __name__ == "__main__":
    main()

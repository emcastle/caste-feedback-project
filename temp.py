"""
# the args expected for each parser
# python -m Caste_Project.parse.parse_pdf -h
usage: parse_pdf.py [-h] --in_dir IN_DIR --out_dir OUT_DIR [--docs_dir DOCS_DIR] [--documents_name DOCUMENTS_NAME]

options:
  -h, --help            show this help message and exit
  --in_dir IN_DIR
  --out_dir OUT_DIR
  --docs_dir DOCS_DIR   Folder containing pdf_documents.parquet (recommended)
  --documents_name DOCUMENTS_NAME

usage: parse_docx.py [-h] --in_dir IN_DIR --out_dir OUT_DIR [--entries_name ENTRIES_NAME] [--docs_dir DOCS_DIR]
                     [--documents_name DOCUMENTS_NAME]

options:
  -h, --help            show this help message and exit
  --in_dir IN_DIR
  --out_dir OUT_DIR
  --entries_name ENTRIES_NAME
  --docs_dir DOCS_DIR   Folder containing docx_documents.parquet (recommended)
  --documents_name DOCUMENTS_NAME

usage: parse_pptx.py [-h] --in_dir IN_DIR --out_dir OUT_DIR [--entries_name ENTRIES_NAME] [--docs_dir DOCS_DIR]
                     [--documents_name DOCUMENTS_NAME] [--strip_anchor_lines]

options:
  -h, --help            show this help message and exit
  --in_dir IN_DIR
  --out_dir OUT_DIR
  --entries_name ENTRIES_NAME
  --docs_dir DOCS_DIR   Folder containing pptx_documents.parquet (recommended)
  --documents_name DOCUMENTS_NAME
  --strip_anchor_lines  Remove From/To/Subject/Date lines from feedback_text


usage: parse_json.py [-h] --in_dir IN_DIR --out_dir OUT_DIR [--entries_name ENTRIES_NAME]
                     [--docs_dir DOCS_DIR] [--documents_name DOCUMENTS_NAME]

options:
  -h, --help            show this help message and exit
  --in_dir IN_DIR
  --out_dir OUT_DIR
  --entries_name ENTRIES_NAME
  --docs_dir DOCS_DIR   Folder containing json_documents.parquet (recommended)
  --documents_name DOCUMENTS_NAME

"""
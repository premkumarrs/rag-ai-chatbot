"""Parser registry."""

from __future__ import annotations

from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.csv import CSVParser
from app.ingestion.parsers.docx import DOCXParser
from app.ingestion.parsers.html import HTMLParser
from app.ingestion.parsers.image import ImageParser
from app.ingestion.parsers.json import JSONParser
from app.ingestion.parsers.markdown import MarkdownParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.pptx import PPTXParser
from app.ingestion.parsers.txt import TXTParser
from app.ingestion.parsers.xlsx import XLSXParser

PARSERS: list[BaseParser] = [
    PDFParser(),
    DOCXParser(),
    TXTParser(),
    MarkdownParser(),
    CSVParser(),
    XLSXParser(),
    PPTXParser(),
    HTMLParser(),
    JSONParser(),
    ImageParser(),
]

EXTENSION_TO_PARSER: dict[str, BaseParser] = {}
for parser in PARSERS:
    for extension in parser.extensions:
        EXTENSION_TO_PARSER[extension] = parser

SUPPORTED_EXTENSIONS = set(EXTENSION_TO_PARSER.keys())

"""Ingestion summary reporting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IngestionReport:
    files_discovered: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_unchanged: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    ocr_pages: int = 0
    ocr_failures: int = 0
    duration_seconds: float = 0.0
    by_type: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def record_type(self, file_type: str) -> None:
        self.by_type[file_type] = self.by_type.get(file_type, 0) + 1

    def print_summary(self) -> None:
        print("=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Files discovered:       {self.files_discovered}")
        print(f"Files processed:        {self.files_processed}")
        print(f"Files skipped:          {self.files_skipped}")
        print(f"Files unchanged:        {self.files_unchanged}")
        print(f"Files failed:           {self.files_failed}")
        print("")
        print("By type:")
        if self.by_type:
            for file_type in sorted(self.by_type):
                print(f"  {file_type.upper():<6} {self.by_type[file_type]}")
        else:
            print("  (none)")
        print("")
        print("OCR:")
        print(f"  OCR pages:            {self.ocr_pages}")
        print(f"  OCR failures:         {self.ocr_failures}")
        print("")
        print(f"Chunks created:         {self.chunks_created}")
        print(f"Embeddings created:     {self.embeddings_created}")
        print(f"Duration:               {self.duration_seconds:.1f} seconds")
        print("=" * 50)
        if self.failures:
            print("Failures:")
            for failure in self.failures:
                print(f"  - {failure}")

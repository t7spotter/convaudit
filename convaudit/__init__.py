"""convaudit: measure what annotation-format converters change.

The harness converts a COCO instances file with third-party tools, reads the
results back with its own small parsers (never the tools' readers), matches
each converted object to its original, and reports what changed.
"""

__version__ = "0.1.0"

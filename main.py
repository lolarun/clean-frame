"""Clean Frame - remove hardcoded subtitles from videos and extract them as SRT.

Usage: python main.py <videos or directories> [-o output] [-m lama|propainter]
"""
import sys

from cli import main

if __name__ == "__main__":
    sys.exit(main())

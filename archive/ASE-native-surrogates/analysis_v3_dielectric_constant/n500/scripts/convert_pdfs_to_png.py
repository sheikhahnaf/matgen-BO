"""
Convert all PDF figures to PNG for PowerPoint compatibility
"""

import subprocess
from pathlib import Path
import os

def convert_pdf_to_png(pdf_path, png_path, dpi=300):
    """Convert a PDF to PNG using pdftoppm (from poppler-utils)."""
    try:
        # Try using pdftoppm (macOS via homebrew poppler)
        cmd = [
            'pdftoppm',
            '-png',
            '-singlefile',
            '-r', str(dpi),
            str(pdf_path),
            str(png_path.with_suffix(''))
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"  pdftoppm failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("  pdftoppm not found. Trying convert (ImageMagick)...")

        try:
            # Try using ImageMagick's convert
            cmd = [
                'convert',
                '-density', str(dpi),
                '-quality', '100',
                str(pdf_path),
                str(png_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return True
            else:
                print(f"  convert failed: {result.stderr}")
                return False

        except FileNotFoundError:
            print("  convert not found. Trying sips (macOS native)...")

            try:
                # Try using macOS's sips
                cmd = [
                    'sips',
                    '-s', 'format', 'png',
                    str(pdf_path),
                    '--out', str(png_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    return True
                else:
                    print(f"  sips failed: {result.stderr}")
                    return False

            except FileNotFoundError:
                print("  ERROR: No PDF conversion tool found!")
                print("  Please install one of: poppler-utils, ImageMagick, or use macOS sips")
                return False

def main():
    """Convert all PDFs in figures/ to PNG in figures_png/."""

    script_dir = Path(__file__).parent
    figures_dir = script_dir.parent / 'figures'
    png_dir = script_dir.parent / 'figures_png'

    print("=" * 80)
    print("CONVERTING PDFs TO PNG FOR POWERPOINT")
    print("=" * 80)

    # Find all PDFs
    pdf_files = list(figures_dir.rglob('*.pdf'))
    print(f"\nFound {len(pdf_files)} PDF files")

    if not pdf_files:
        print("No PDF files found!")
        return

    # Create PNG directory structure
    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        # Create corresponding PNG path
        rel_path = pdf_path.relative_to(figures_dir)
        png_path = png_dir / rel_path.with_suffix('.png')

        # Create subdirectory if needed
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert
        print(f"\n  Converting: {rel_path}")

        if convert_pdf_to_png(pdf_path, png_path):
            print(f"    ✓ Created: {png_path.relative_to(script_dir.parent)}")
            success_count += 1
        else:
            print(f"    ✗ Failed: {rel_path}")
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"Conversion complete:")
    print(f"  Success: {success_count}/{len(pdf_files)}")
    print(f"  Failed: {fail_count}/{len(pdf_files)}")
    print(f"  PNG directory: {png_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

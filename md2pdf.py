#!/usr/bin/env python3
"""
Markdown to PDF (VSCode-preview-like) via:
- python-markdown
- Pygments (nice code highlighting + theme CSS)
- matplotlib.mathtext (render LaTeX math to SVG, no JS needed)
- WeasyPrint
```
pip install markdown pygments matplotlib weasyprint==59.0 pydyf==0.8.0
pip install latex2mathml
```
Usage:
    python md2pdf.py input.md [output.pdf]
    python md2pdf.py hw1.md --pygments-style vs
    python md2pdf.py hw1.md --pygments-style friendly
    python md2pdf.py hw1.md --pygments-style monokai
"""

import sys
import os
import re
import argparse
import base64
import html as html_escape
from io import BytesIO
from typing import Dict, Tuple

import markdown
from weasyprint import HTML

from pygments.formatters import HtmlFormatter
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties


# -------------------------
# Math rendering (LaTeX -> SVG data URI)
# -------------------------
_math_cache: Dict[Tuple[str, int], str] = {}

def _tex_to_svg_data_uri(tex: str, fontsize: int) -> str:
    """Render TeX (math-mode content, WITHOUT $) into an SVG data URI."""
    key = (tex, fontsize)
    if key in _math_cache:
        return _math_cache[key]

    buf = BytesIO()
    prop = FontProperties(size=fontsize)

    # matplotlib mathtext expects a math expression with $...$
    mathtext.math_to_image(f"${tex}$", buf, format="svg", prop=prop, dpi=200)
    svg_bytes = buf.getvalue()
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    uri = f"data:image/svg+xml;base64,{b64}"
    _math_cache[key] = uri
    return uri


# -------------------------
# Protect fenced code blocks so math regex won't touch them
# -------------------------
_FENCE_RE = re.compile(r'^(\s*)(```|~~~)')

def _protect_fenced_code(md_text: str):
    """Replace fenced code blocks with placeholders to avoid math replacement inside them."""
    lines = md_text.splitlines(keepends=True)
    out = []
    blocks = {}
    in_fence = False
    fence = None
    buf = []
    key = None
    idx = 0

    for line in lines:
        if not in_fence:
            m = _FENCE_RE.match(line)
            if m:
                in_fence = True
                fence = m.group(2)  # ``` or ~~~
                buf = [line]
                key = f"@@FENCED_CODE_BLOCK_{idx}@@"
                idx += 1
            else:
                out.append(line)
        else:
            buf.append(line)
            # end fence: line starts with the same fence marker
            if line.lstrip().startswith(fence):
                in_fence = False
                blocks[key] = "".join(buf)
                out.append(key + "\n")
                buf = []
                fence = None
                key = None

    # If unclosed fence, just keep it as-is
    if in_fence and buf:
        out.extend(buf)

    return "".join(out), blocks


def _restore_fenced_code(md_text: str, blocks: Dict[str, str]):
    for k, v in blocks.items():
        md_text = md_text.replace(k, v)
    return md_text


# -------------------------
# Math replacement: $$...$$ and $...$
# -------------------------
BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Inline math: $...$  (avoid $$...$$ already handled; avoid \$)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)([^\n]+?)(?<!\\)\$")

def _render_math_in_markdown(md_text: str) -> str:
    """Replace math delimiters with <img> tags pointing to SVG."""
    protected, blocks = _protect_fenced_code(md_text)

    def repl_block(m):
        tex = m.group(1).strip()
        src = _tex_to_svg_data_uri(tex, fontsize=16)
        alt = html_escape.escape(tex, quote=True)
        return f'\n<div class="math-block"><img class="math-display" src="{src}" alt="{alt}"></div>\n'

    def repl_inline(m):
        tex = m.group(1).strip()
        src = _tex_to_svg_data_uri(tex, fontsize=12)
        alt = html_escape.escape(tex, quote=True)
        return f'<img class="math-inline" src="{src}" alt="{alt}">'

    protected = BLOCK_MATH_RE.sub(repl_block, protected)
    protected = INLINE_MATH_RE.sub(repl_inline, protected)

    return _restore_fenced_code(protected, blocks)


def convert_md_to_pdf(input_file: str, output_file: str = None, pygments_style: str = "xcode"):
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.pdf"

    with open(input_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 1) Pre-render math into SVG <img>
    md_content = _render_math_in_markdown(md_content)

    # 2) Markdown -> HTML
    html_content = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "noclasses": False,
                "pygments_style": pygments_style,
            }
        },
    )

    # 3) Pygments CSS (critical for nice code blocks)
    pyg_css = HtmlFormatter(style=pygments_style).get_style_defs(".codehilite")

    css_style = f"""
    @page {{
        size: A4;
        margin: 2cm;
        
        /* Page Numbers */
        @bottom-left {{
            content: counter(page) " / " counter(pages);
            font-family: -apple-system, system-ui, sans-serif;
            font-size: 9pt;
            color: #6e7781;
        }}
    }}

    /* Body typography close to VSCode/GitHub markdown */
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei",
                     "WenQuanYi Micro Hei", Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #24292f;
    }}

    h1 {{
        font-size: 22pt;
        border-bottom: 2px solid #d0d7de;
        padding-bottom: 8px;
        margin-top: 28px;
    }}
    h2 {{
        font-size: 16pt;
        border-bottom: 1px solid #d0d7de;
        padding-bottom: 4px;
        margin-top: 22px;
    }}
    h3 {{ font-size: 13pt; margin-top: 18px; }}
    h4 {{ font-size: 12pt; margin-top: 14px; }}

    /* Inline code */
    code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 10pt;
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 4px;
        padding: 0.1em 0.35em;
    }}

    /* Code blocks (VSCode/GitHub-like) */
    .codehilite {{
        margin: 14px 0;
    }}
    .codehilite pre {{
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 12px 14px;

        /* Fix (3): prevent clipping in PDF */
        white-space: pre-wrap;            /* allow wrapping */
        overflow-wrap: anywhere;          /* break long tokens */
        word-break: break-word;

        line-height: 1.45;
        font-size: 9.5pt;
    }}
    .codehilite pre code {{
        background: transparent;
        border: none;
        padding: 0;
        white-space: inherit;
    }}

    /* Math */
    .math-block {{
        text-align: center;
        margin: 10px 0 14px 0;
    }}
    img.math-display {{
        max-width: 100%;
        height: auto;
    }}
    img.math-inline {{
        height: 1.15em;
        vertical-align: -0.15em;
    }}

    /* Tables */
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0;
    }}
    th, td {{
        border: 1px solid #d0d7de;
        padding: 8px 10px;
        text-align: left;
    }}
    th {{
        background: #f6f8fa;
        font-weight: 600;
    }}

    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 12px auto;
    }}

    blockquote {{
        border-left: 4px solid #d0d7de;
        margin: 12px 0;
        padding: 8px 12px;
        background: #f6f8fa;
        color: #57606a;
    }}

    a {{ color: #0969da; text-decoration: none; }}

    /* Avoid ugly breaks */
    pre, table, blockquote {{
        break-inside: avoid;
    }}

    /* Pygments theme */
    {pyg_css}
    """

    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>{css_style}</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    base_url = os.path.dirname(os.path.abspath(input_file))
    HTML(string=full_html, base_url=base_url).write_pdf(output_file)

    print(f"✓ PDF created successfully: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF (VSCode-like, with math)")
    parser.add_argument("input", help="Input Markdown file path")
    parser.add_argument("output", nargs="?", help="Output PDF file path (optional)")
    parser.add_argument("--pygments-style", default="xcode", help="Pygments style (e.g., xcode, vs, friendly, monokai)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    try:
        convert_md_to_pdf(args.input, args.output, pygments_style=args.pygments_style)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
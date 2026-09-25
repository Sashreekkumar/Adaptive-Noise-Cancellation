# -*- coding: utf-8 -*-
"""
build_docs_data.py
Official, clean documentation parser for noisygen.
Produces clean HTML, standard technical callouts, and search indexing.
"""

import os
import re
import html
import json

def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text.lower()).strip()
    return re.sub(r'[-\s]+', '-', text)

def build_data():
    with open('readme.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into sections based on "# " or "# [0-9]+\. "
    raw_sections = re.split(r'\n(?=# [0-9]+\. )', content)
    
    sections_data = []

    group_mapping = {
        0: ("Overview", 1),
        1: ("Getting Started", 2),
        2: ("Getting Started", 2),
        3: ("Getting Started", 2),
        44: ("Getting Started", 2),
        
        4: ("CLI Reference", 3),
        45: ("CLI Reference", 3),
        
        5: ("Configuration", 4),
        6: ("Configuration", 4),
        7: ("Configuration", 4),
        8: ("Configuration", 4),
        9: ("Configuration", 4),
        10: ("Configuration", 4),
        11: ("Configuration", 4),
        12: ("Configuration", 4),
        13: ("Configuration", 4),
        14: ("Configuration", 4),
        15: ("Configuration", 4),
        16: ("Configuration", 4),
        17: ("Configuration", 4),
        18: ("Configuration", 4),
        19: ("Configuration", 4),
        20: ("Configuration", 4),
        21: ("Configuration", 4),
        22: ("Configuration", 4),
        23: ("Configuration", 4),
        24: ("Configuration", 4),
        25: ("Configuration", 4),
        26: ("Configuration", 4),
        
        27: ("Dataset Output & Schemas", 5),
        28: ("Dataset Output & Schemas", 5),
        29: ("Dataset Output & Schemas", 5),
        30: ("Dataset Output & Schemas", 5),
        31: ("Dataset Output & Schemas", 5),
        32: ("Dataset Output & Schemas", 5),
        33: ("Dataset Output & Schemas", 5),
        34: ("Dataset Output & Schemas", 5),
        35: ("Dataset Output & Schemas", 5),
        36: ("Dataset Output & Schemas", 5),
        
        37: ("Audio Processing & Math", 6),
        38: ("Audio Processing & Math", 6),
        39: ("Audio Processing & Math", 6),
        40: ("Audio Processing & Math", 6),
        41: ("Audio Processing & Math", 6),
        42: ("Audio Processing & Math", 6),
        43: ("Audio Processing & Math", 6),
        
        46: ("Architecture & Data Flow", 7),
        47: ("Architecture & Data Flow", 7),
        48: ("Architecture & Data Flow", 7),
    }

    for idx, sec_text in enumerate(raw_sections):
        sec_text = sec_text.strip()
        if not sec_text:
            continue
            
        lines = sec_text.split('\n')
        h1_line = lines[0].strip()
        body_lines = lines[1:]
        
        m = re.match(r'^#\s*(?:([0-9]+)\.\s*)?(.*)$', h1_line)
        if m:
            sec_num = int(m.group(1)) if m.group(1) else 0
            title = m.group(2).strip()
        else:
            sec_num = idx
            title = h1_line.replace('#', '').strip()
            
        group_title, group_order = group_mapping.get(sec_num, ("General Reference", 99))
        
        if sec_num == 0:
            slug = "overview"
        else:
            slug = f"sec-{sec_num:02d}-" + slugify(title)
            
        clean_title = title.replace('`', '')
        
        toc = []
        html_content, plain_text = render_section_content(sec_num, title, body_lines, toc, slug)
        
        summary = ""
        clean_paras = [p.strip() for p in plain_text.split('\n') if len(p.strip()) > 30 and not p.strip().startswith('```')]
        if clean_paras:
            summary = clean_paras[0][:160] + ('...' if len(clean_paras[0]) > 160 else '')
        else:
            summary = f"Technical documentation and specifications for {clean_title}."

        sections_data.append({
            "num": sec_num,
            "id": slug,
            "title": clean_title,
            "rawTitle": title,
            "group": group_title,
            "groupOrder": group_order,
            "toc": toc,
            "summary": summary,
            "html": html_content,
            "plainText": plain_text[:2000]
        })

    def sort_key(s):
        num = s["num"]
        if num == 0:
            return (1, 0)
        elif num == 44:
            return (2, 44)
        elif num == 45:
            return (3, 99)
        else:
            return (s["groupOrder"], num)

    sections_data.sort(key=sort_key)
    return sections_data

def render_section_content(sec_num, main_title, body_lines, toc, section_slug):
    raw_text = '\n'.join(body_lines).strip()
    plain_text = re.sub(r'```.*?```', '', raw_text, flags=re.DOTALL)
    plain_text = re.sub(r'#+\s*', '', plain_text)
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()

    processed_blocks = []

    # Section 0 Official Package Header
    if sec_num == 0:
        processed_blocks.append('''
<div class="package-meta-header">
  <div class="meta-badges-row">
    <span class="meta-tag">Python &ge; 3.8</span>
    <span class="meta-tag">License: MIT / Apache</span>
    <span class="meta-tag">Status: Production</span>
    <span class="meta-tag">Standard: 48 kHz Mono PCM_16</span>
  </div>
  <p class="package-lead-desc">
    <code>noisygen</code> is a reproducible, provenance-preserving synthetic noisy-speech dataset generator for speech enhancement, automatic speech recognition (ASR), and audio machine learning pipelines.
  </p>
</div>
''')

    # Section 40 Clean Math Display
    if sec_num == 40:
        processed_blocks.append('''
<div class="official-math-box">
  <div class="math-box-title">Mathematical Formulation</div>
  <div class="math-equation">
    $$\\text{SNR}_{\\text{dB}} = 20 \\log_{10}\\left(\\frac{\\text{RMS}_{\\text{clean}}}{\\text{RMS}_{\\text{noise}}}\\right)$$
  </div>
  <div class="math-equation">
    $$G_{\\text{noise}} = \\frac{\\text{RMS}_{\\text{clean}}}{\\text{RMS}_{\\text{noise}} \\cdot 10^{\\frac{\\text{target\\_snr}}{20}}}$$
  </div>
  <p class="math-caption">
    The composite noise signal is multiplied by scalar factor <strong>G</strong> and summed with the clean speech signal:
    <code>noisy = clean + G * composite_noise</code>.
  </p>
</div>
''')

    # Section 45 Clean Exit Codes Table
    if sec_num == 45:
        processed_blocks.append('''
<div class="table-container">
  <table class="standard-table">
    <thead>
      <tr>
        <th style="width: 100px;">Exit Code</th>
        <th style="width: 160px;">Status</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>0</code></td>
        <td><strong>Success</strong></td>
        <td>Generation or verification completed successfully. All outputs and metadata are validated.</td>
      </tr>
      <tr>
        <td><code>1</code></td>
        <td><strong>Failure</strong></td>
        <td>Sample generation failed or integrity check failed. Inspect <code>failed_attempts.jsonl</code>.</td>
      </tr>
      <tr>
        <td><code>2</code></td>
        <td><strong>Configuration Error</strong></td>
        <td>Invalid CLI arguments, schema validation failure, or missing source directories.</td>
      </tr>
      <tr>
        <td><code>130</code></td>
        <td><strong>Interrupted</strong></td>
        <td>Process received SIGINT (Ctrl-C). Partial progress is safely resumable with <code>--resume</code>.</td>
      </tr>
    </tbody>
  </table>
</div>
''')

    # Section 47 Clean Data Transformation Diagram
    if sec_num == 47:
        processed_blocks.append('''
<div class="pipeline-diagram-box">
  <div class="pipeline-diagram-title">Data Transformation Flow</div>
  <div class="pipeline-diagram-steps">
    <div class="diagram-step">
      <span class="step-badge">Stage 1</span>
      <strong>Source Ingestion</strong>
      <span class="step-sub">Scan wav/ and noise/ trees</span>
    </div>
    <div class="diagram-separator">&rarr;</div>
    <div class="diagram-step">
      <span class="step-badge">Stage 2</span>
      <strong>Standardisation</strong>
      <span class="step-sub">Resample to 48 kHz mono</span>
    </div>
    <div class="diagram-separator">&rarr;</div>
    <div class="diagram-step">
      <span class="step-badge">Stage 3</span>
      <strong>Clean Crop</strong>
      <span class="step-sub">Speech activity detection</span>
    </div>
    <div class="diagram-separator">&rarr;</div>
    <div class="diagram-step">
      <span class="step-badge">Stage 4</span>
      <strong>Noise Composite</strong>
      <span class="step-sub">Category rules & gain mix</span>
    </div>
    <div class="diagram-separator">&rarr;</div>
    <div class="diagram-step">
      <span class="step-badge">Stage 5</span>
      <strong>SNR Scaling</strong>
      <span class="step-sub">Calculate & apply gain G</span>
    </div>
    <div class="diagram-separator">&rarr;</div>
    <div class="diagram-step">
      <span class="step-badge">Stage 6</span>
      <strong>Anti-Clipping</strong>
      <span class="step-sub">Scale mix or reject</span>
    </div>
    <div class="diagram-separator">&rarr;</div>
    <div class="diagram-step">
      <span class="step-badge">Stage 7</span>
      <strong>Two-Phase Commit</strong>
      <span class="step-sub">Write WAV, append JSONL</span>
    </div>
  </div>
</div>
''')

    i = 0
    total = len(body_lines)
    
    while i < total:
        line = body_lines[i]
        stripped = line.strip()
        
        if not stripped:
            i += 1
            continue
            
        # Code block
        if stripped.startswith('```'):
            lang_match = re.match(r'^```([a-zA-Z0-9_-]*)', stripped)
            lang = lang_match.group(1).strip() if lang_match else 'text'
            if not lang:
                lang = 'text'
            code_lines = []
            i += 1
            while i < total and not body_lines[i].strip().startswith('```'):
                code_lines.append(body_lines[i])
                i += 1
            i += 1
            
            code_str = '\n'.join(code_lines).strip('\n')
            if lang == 'text' and ('├──' in code_str or '└──' in code_str or 'wav/' in code_str):
                lang = 'tree'
            elif lang == 'text' and ('--' in code_str or 'python -m' in code_str):
                lang = 'bash'
            elif lang == 'text' and (code_str.startswith('{') or code_str.startswith('[')):
                lang = 'json'
                
            escaped_code = html.escape(code_str)
            styled_code = highlight_code(escaped_code, lang)
            
            processed_blocks.append(f'''<div class="standard-code-block" data-lang="{lang}">
  <div class="code-block-header">
    <span class="code-lang-label">{lang.upper()}</span>
    <button class="code-copy-button" onclick="copyCode(this)" title="Copy to clipboard">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      <span class="copy-text">Copy</span>
    </button>
  </div>
  <pre><code class="language-{lang}">{styled_code}</code></pre>
</div>''')
            continue
            
        # Headings
        if stripped.startswith('### '):
            h_text = stripped[4:].strip()
            h_clean = h_text.replace('`', '')
            sub_id = section_slug + "-" + slugify(h_clean)
            toc.append({"id": sub_id, "text": h_clean, "level": 3})
            processed_blocks.append(f'<h3 id="{sub_id}" class="standard-h3"><a href="#{sub_id}" class="anchor-link">#</a>{format_inline(h_text)}</h3>')
            i += 1
            continue
            
        if stripped.startswith('## '):
            h_text = stripped[3:].strip()
            h_clean = h_text.replace('`', '')
            sub_id = section_slug + "-" + slugify(h_clean)
            toc.append({"id": sub_id, "text": h_clean, "level": 2})
            processed_blocks.append(f'<h2 id="{sub_id}" class="standard-h2"><a href="#{sub_id}" class="anchor-link">#</a>{format_inline(h_text)}</h2>')
            i += 1
            continue

        # Horizontal Rule
        if stripped in ('---', '***', '___'):
            processed_blocks.append('<hr class="section-divider" />')
            i += 1
            continue

        # Numbered Principles
        if re.match(r'^[0-9]+\.\s+\*\*', stripped):
            items = []
            while i < total and re.match(r'^[0-9]+\.\s+', body_lines[i].strip()):
                item_line = body_lines[i].strip()
                item_content = re.sub(r'^[0-9]+\.\s+', '', item_line)
                items.append(f'<li>{format_inline(item_content)}</li>')
                i += 1
            processed_blocks.append(f'<ol class="official-numbered-list">{"".join(items)}</ol>')
            continue

        # Bullet list
        if stripped.startswith('- ') or stripped.startswith('* '):
            items = []
            while i < total and (body_lines[i].strip().startswith('- ') or body_lines[i].strip().startswith('* ')):
                item_text = body_lines[i].strip()[2:].strip()
                items.append(f'<li>{format_inline(item_text)}</li>')
                i += 1
            processed_blocks.append(f'<ul class="standard-list">{"".join(items)}</ul>')
            continue

        # Collect paragraph
        para_lines = []
        while i < total and body_lines[i].strip() and not body_lines[i].strip().startswith('```') and not body_lines[i].strip().startswith('#') and not body_lines[i].strip().startswith('- ') and not body_lines[i].strip().startswith('* ') and not re.match(r'^[0-9]+\.\s+\*\*', body_lines[i].strip()) and body_lines[i].strip() not in ('---', '***'):
            para_lines.append(body_lines[i].strip())
            i += 1
            
        full_p = ' '.join(para_lines)
        if full_p:
            if 'The generator is designed around four principles' in full_p:
                processed_blocks.append(f'<p class="standard-p" style="font-weight: 500;">{format_inline(full_p)}</p>')
            elif 'IMPORTANT' in full_p.upper() or 'MUST' in full_p:
                processed_blocks.append(f'<div class="alert-box alert-important"><div class="alert-label">IMPORTANT</div><div class="alert-content">{format_inline(full_p)}</div></div>')
            elif 'silent and the attempt is rejected' in full_p or 'failed attempts are recorded' in full_p.lower():
                processed_blocks.append(f'<div class="alert-box alert-warning"><div class="alert-label">WARNING</div><div class="alert-content">{format_inline(full_p)}</div></div>')
            elif full_p.startswith('Default:'):
                processed_blocks.append(f'<div class="default-row"><span class="default-label">Default:</span> <code>{format_inline(full_p[8:].strip())}</code></div>')
            elif full_p.startswith('Example:'):
                processed_blocks.append(f'<p class="example-lead"><strong>Example:</strong></p>')
            else:
                processed_blocks.append(f'<p class="standard-p">{format_inline(full_p)}</p>')

    return '\n'.join(processed_blocks), plain_text

def format_inline(text):
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`(.+?)`', r'<code class="inline-code">\1</code>', text)
    text = text.replace('—', '&mdash;').replace('--', '&mdash;')
    return text

def highlight_code(code_escaped, lang):
    if lang == 'bash':
        code_escaped = re.sub(r'(--[a-zA-Z0-9_-]+)', r'<span class="code-flag">\1</span>', code_escaped)
        code_escaped = re.sub(r'\b(python|python3|pip)\b', r'<span class="code-kw">\1</span>', code_escaped)
        code_escaped = re.sub(r'\b(noisygen|generate_dataset\.py)\b', r'<span class="code-target">\1</span>', code_escaped)
        return code_escaped
    elif lang == 'json':
        code_escaped = re.sub(r'(&quot;.*?&quot;)(\s*:)', r'<span class="code-key">\1</span>\2', code_escaped)
        code_escaped = re.sub(r':\s*([0-9.-]+)\b', r': <span class="code-num">\1</span>', code_escaped)
        code_escaped = re.sub(r':\s*(true|false|null)\b', r': <span class="code-bool">\1</span>', code_escaped)
        return code_escaped
    elif lang == 'tree':
        code_escaped = re.sub(r'([│├──└───]+)', r'<span class="code-tree">\1</span>', code_escaped)
        code_escaped = re.sub(r'([a-zA-Z0-9_\-\.]+\.wav)', r'<span class="code-wav">\1</span>', code_escaped)
        code_escaped = re.sub(r'([a-zA-Z0-9_\-\.]+\.json[l]?)', r'<span class="code-json">\1</span>', code_escaped)
        code_escaped = re.sub(r'([a-zA-Z0-9_\-\.]+/)\b', r'<span class="code-dir">\1</span>', code_escaped)
        return code_escaped
    return code_escaped

if __name__ == '__main__':
    data = build_data()
    print(f'Parsed {len(data)} sections with official documentation formatting.')
    js_output = "window.DOCS_DATA = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n"
    with open('docs-data.js', 'w', encoding='utf-8') as f:
        f.write(js_output)
    print('Updated docs-data.js successfully!')

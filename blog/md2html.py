#!/usr/bin/env python3
"""Minimal Markdown -> standalone offline HTML for the blog drafts (headings, paragraphs, nested - lists, 1. lists,
pipe tables, **bold**, *italic*, `code`, [links](url)). Styles come from page-head.html (fonts made local-only).
usage: md2html.py IN.md OUT.html"""
import html, re, sys

src, dst = sys.argv[1:3]
lines = open(src).read().split("\n")


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?!\w)", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


out, i = [], 0
title = None
while i < len(lines):
    l = lines[i]
    if not l.strip():
        i += 1
        continue
    if l.startswith("# "):
        title = l[2:].strip()
        out.append(f"<h1>{inline(title)}</h1>")
        i += 1
    elif l.startswith("## "):
        out.append(f"<h2>{inline(l[3:].strip())}</h2>")
        i += 1
    elif l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
        t = ["<div class=\"tbl\"><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
        t += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body]
        out.append("".join(t) + "</tbody></table></div>")
    elif re.match(r"^( *)(- |\d+\. )", l):
        # nested lists by indentation (2 spaces per level for '-', 3 for '1.')
        stack = []  # (indent, tag)
        items = []
        while i < len(lines) and (re.match(r"^( *)(- |\d+\. )", lines[i]) or
                                  (lines[i].startswith("  ") and lines[i].strip() and stack)):
            m = re.match(r"^( *)(- |\d+\. )(.*)", lines[i])
            if not m:  # continuation line of the previous item
                items[-1] = items[-1][:-5] + " " + inline(lines[i].strip()) + "</li>"
                i += 1
                continue
            ind, mark, text = len(m.group(1)), m.group(2), m.group(3)
            tag = "ul" if mark == "- " else "ol"
            while stack and ind < stack[-1][0]:
                items.append(f"</{stack.pop()[1]}></li>")
            if not stack or ind > stack[-1][0]:
                if stack and items and items[-1].endswith("</li>"):
                    items[-1] = items[-1][:-5]
                stack.append((ind, tag))
                items.append(f"<{tag}>")
            items.append(f"<li>{inline(text)}</li>")
            i += 1
        while stack:
            t = stack.pop()[1]
            items.append(f"</{t}>" + ("</li>" if stack else ""))
        out.append("".join(items))
    else:
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||( *)(- |\d+\. ))", lines[i]):
            para.append(lines[i].strip())
            i += 1
        p = inline(" ".join(para))
        cls = ' class="note"' if para and para[0].startswith("*") and para[-1].endswith("*") else ""
        out.append(f"<p{cls}>{p}</p>")

head = open(__file__.rsplit("/", 1)[0] + "/page-head.html").read()
head = re.sub(r'<link rel="preconnect"[^>]*>\n', "", head)
head = re.sub(r'<link rel="stylesheet" href="https://fonts.googleapis.com[^>]*>\n', "", head)
head = head.replace('"Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif', '"Helvetica Neue", "Segoe UI", Arial, sans-serif')
head = head.replace('"Source Serif 4", Georgia, "Times New Roman", serif', 'Charter, "Bitstream Charter", Georgia, "Times New Roman", serif')
head = head.replace('"IBM Plex Mono", ui-monospace,', "ui-monospace,")
head = re.sub(r"<title>.*?</title>", f"<title>{html.escape(re.sub(r'[*`]', '', title or 'Draft'))}</title>", head)
doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
       "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" + head +
       "</head>\n<body>\n<article class=\"wrap\">\n" + "\n".join(out) + "\n</article>\n</body>\n</html>\n")
assert "googleapis" not in doc
open(dst, "w").write(doc)
print(dst, len(doc))

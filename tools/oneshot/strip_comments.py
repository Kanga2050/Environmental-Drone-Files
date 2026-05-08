
import io
import re
import sys
import tokenize

IDENT_END = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$)]}")

def strip_js(src):
    out = []
    i, n = 0, len(src)
    prev = ""
    while i < n:
        c = src[i]
        two = src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/*":
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                if src[i] == "\\":
                    out.append(src[i:i + 2])
                    i += 2
                    continue
                out.append(src[i])
                if src[i] == q:
                    i += 1
                    break
                i += 1
            prev = q
            continue
        if c == "`":
            out.append(c)
            i += 1
            depth = 0
            while i < n:
                if src[i] == "\\":
                    out.append(src[i:i + 2])
                    i += 2
                    continue
                if src[i:i + 2] == "${":
                    depth += 1
                    out.append("${")
                    i += 2
                    continue
                if depth and src[i] == "}":
                    depth -= 1
                    out.append("}")
                    i += 1
                    continue
                out.append(src[i])
                if src[i] == "`" and depth == 0:
                    i += 1
                    break
                i += 1
            prev = "`"
            continue
        if c == "/" and prev not in IDENT_END:
            out.append(c)
            i += 1
            inclass = False
            while i < n:
                if src[i] == "\\":
                    out.append(src[i:i + 2])
                    i += 2
                    continue
                if src[i] == "[":
                    inclass = True
                elif src[i] == "]":
                    inclass = False
                out.append(src[i])
                if src[i] == "/" and not inclass:
                    i += 1
                    break
                i += 1
            prev = "/"
            continue
        out.append(c)
        if not c.isspace():
            prev = c
        i += 1
    return "".join(out)

def strip_py(src, drop_docstrings=False):
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    lines = src.splitlines(True)
    cuts = []
    for t in toks:
        if t.type == tokenize.COMMENT:
            cuts.append((t.start, t.end))
    if drop_docstrings:
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.FunctionDef,
                                     ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body = getattr(node, "body", [])
            if (len(body) > 1 and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                d = body[0]
                cuts.append(((d.lineno, d.col_offset),
                             (d.end_lineno, d.end_col_offset)))
    for (sl, sc), (el, ec) in sorted(cuts, reverse=True):
        if sl == el:
            lines[sl - 1] = lines[sl - 1][:sc] + lines[sl - 1][ec:]
        else:
            lines[sl - 1] = lines[sl - 1][:sc] + lines[el - 1][ec:]
            del lines[sl:el]
    return "".join(lines)

def strip_css(src):
    return re.sub(r"/\*.*?\*/", "", src, flags=re.S)

def strip_html(src):
    return re.sub(r"<!--.*?-->", "", src, flags=re.S)

def strip_sh(src):
    out = []
    for k, line in enumerate(src.splitlines(True)):
        if k == 0 and line.startswith("#!"):
            out.append(line)
            continue
        s = line.lstrip()
        if s.startswith("#"):
            continue
        out.append(re.sub(r"\s+#\s.*$", "", line.rstrip("\n")) + "\n")
    return "".join(out)

def tidy(src):
    src = "\n".join(l.rstrip() for l in src.splitlines())
    src = re.sub(r"\n{3,}", "\n\n", src)
    src = re.sub(r"\{\n\n+", "{\n", src)
    return src.rstrip() + "\n"

if __name__ == "__main__":
    path, kind = sys.argv[1], sys.argv[2]
    doc = len(sys.argv) > 3 and sys.argv[3] == "docstrings"
    s = io.open(path, encoding="utf-8").read()
    before = len(s.splitlines())
    fn = {"js": strip_js, "css": strip_css, "html": strip_html,
          "sh": strip_sh}.get(kind)
    s = strip_py(s, doc) if kind == "py" else fn(s)
    s = tidy(s)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print(f"{path}: {before} -> {len(s.splitlines())} lines")

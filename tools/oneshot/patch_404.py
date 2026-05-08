import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
n = 0
for label in ("request", "multi-view request"):
    old = f'''        except Exception as exc:                      # network / quota / auth
            print(f"    Pl@ntNet {label} failed: {{exc}}")
            return []'''
    new = f'''        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return []                                 # nothing matched
            print(f"    Pl@ntNet {label} failed: {{exc}}")
            return []
        except Exception as exc:                      # network / quota / auth
            print(f"    Pl@ntNet {label} failed: {{exc}}")
            return []'''
    if old in s:
        s = s.replace(old, new, 1)
        n += 1
if not n:
    import re
    hits = re.findall(r"except Exception as exc:.*?\n.*?Pl@ntNet.*?\n", s)
    sys.exit(f"no handler matched; found {len(hits)} candidates:\n" + "".join(hits))
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
import ast
ast.parse(s)
print(f"404 now means 'no match', not an error ({n} handlers)")

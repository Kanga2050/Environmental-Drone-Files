
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    def _send(self, body, ctype="application/octet-stream", code=200, cache=False):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()''',
    '''    def _send(self, body, ctype="application/octet-stream", code=200,
              cache=False, etag=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        elif etag:
            # no-cache means "ask me first", not "do not store": the browser
            # keeps the copy and revalidates, so an unchanged file costs a 304
            # and no body, while an edited one is never served stale.
            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", etag)
        self.end_headers()'''))

subs.append((
    '''    def _file(self, path, ctype=None):
        if not os.path.exists(path):
            return self._send("not found", "text/plain", 404)
        with open(path, "rb") as f:
            body = f.read()
        ctype = ctype or (mimetypes.guess_type(path)[0] or "application/octet-stream")
        self._send(body, ctype, cache=path.endswith((".js", ".css", ".jpg")))''',
    '''    def _file(self, path, ctype=None):
        if not os.path.exists(path):
            return self._send("not found", "text/plain", 404)
        st = os.stat(path)
        # Only frame photos are safe to cache hard -- their content is fixed by
        # the frame id. Everything else is source, and is edited while a server
        # is running.
        hard = path.endswith(".jpg")
        etag = None
        if not hard:
            etag = '"%x-%x"' % (int(st.st_mtime), st.st_size)
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", "0")
                return self.end_headers()
        with open(path, "rb") as f:
            body = f.read()
        ctype = ctype or (mimetypes.guess_type(path)[0] or "application/octet-stream")
        self._send(body, ctype, cache=hard, etag=etag)'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("view_map3d: source revalidates, photos stay cached")

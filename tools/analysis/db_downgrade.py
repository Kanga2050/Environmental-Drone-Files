import os
import sqlite3
import sys

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    os.remove(dst)
s = sqlite3.connect(src)
d = sqlite3.connect(dst)
d.executescript("""
CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
  params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
  prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
  prior_tx REAL, prior_ty REAL, prior_tz REAL,
  CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
  FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB);
CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL,
  rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL,
  F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
""")
d.executemany("insert into cameras values (?,?,?,?,?,?)",
              s.execute("select camera_id,model,width,height,params,prior_focal_length from cameras"))
d.executemany("insert into images (image_id,name,camera_id) values (?,?,?)",
              s.execute("select image_id,name,camera_id from images"))
for t in ("keypoints", "matches"):
    d.executemany(f"insert into {t} values (?,?,?,?)",
                  s.execute(f"select * from {t}"))
cols = [r[1] for r in s.execute("pragma table_info(two_view_geometries)")]
pick = ["pair_id", "rows", "cols", "data", "config", "F", "E", "H", "qvec", "tvec"]
sel = ",".join(c if c in cols else "NULL" for c in pick)
d.executemany("insert into two_view_geometries values (?,?,?,?,?,?,?,?,?,?)",
              s.execute(f"select {sel} from two_view_geometries"))
d.commit()
for t in ("cameras", "images", "keypoints", "matches", "two_view_geometries"):
    print(f"{t}: {d.execute(f'select count(*) from {t}').fetchone()[0]:,}")
d.close()
print(f"{os.path.getsize(dst)/1e9:.2f} GB -> {dst}")

import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    ENDPOINT = "https://my-api.plantnet.org/v2/identify/all"

    def __init__(self, api_key, timeout=30):
        self.api_key = api_key
        self.timeout = timeout''',
    '''    BASE = "https://my-api.plantnet.org/v2/identify"
    PROJECT = "k-southwestern-u-s-a"

    def __init__(self, api_key, timeout=30, project=None):
        self.api_key = api_key
        self.timeout = timeout
        self.project = project or self.PROJECT
        self.ENDPOINT = f"{self.BASE}/{self.project}"'''))

subs.append((
    '''    ap.add_argument("--plantnet_api_key", default=None,''',
    '''    ap.add_argument("--plantnet_project", default=PlantNetClassifier.PROJECT,
                    help="Pl@ntNet flora to search. Regional floras score far "
                         "better than 'all' because they exclude species that "
                         "cannot occur where the photo was taken; the default "
                         "is the TDWG region covering California")
    ap.add_argument("--plantnet_api_key", default=None,'''))

subs.append((
    '''        classifier = PlantNetClassifier(args.plantnet_api_key)''',
    '''        classifier = PlantNetClassifier(args.plantnet_api_key,
                                        project=args.plantnet_project)
        print(f"  Pl@ntNet flora: {args.plantnet_project}")'''))

for old, new in subs:
    if old not in s:
        sys.exit("detect_plants NOT FOUND:\n" + old[:200])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("detect_plants: Pl@ntNet scoped to a regional flora")

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = [
    ('''    ap.add_argument("--plantnet_api_key", default=None)''',
     '''    ap.add_argument("--plantnet_api_key", default=None)
    ap.add_argument("--plantnet_project", default="k-southwestern-u-s-a")'''),
    ('''        scene.enable_classifier(args.classify, args.plantnet_api_key,
                                args.models_dir)''',
     '''        scene.enable_classifier(args.classify, args.plantnet_api_key,
                                args.models_dir, args.plantnet_project)'''),
    ('''    def enable_classifier(self, kind, plantnet_key=None, models_dir=None):
        self.classifier_kind = kind
        self._plantnet_key = plantnet_key''',
     '''    def enable_classifier(self, kind, plantnet_key=None, models_dir=None,
                          plantnet_project=None):
        self.classifier_kind = kind
        self._plantnet_key = plantnet_key
        self._plantnet_project = plantnet_project'''),
    ('''                self._classifier = dp.PlantNetClassifier(self._plantnet_key)''',
     '''                self._classifier = dp.PlantNetClassifier(
                    self._plantnet_key, project=self._plantnet_project)'''),
]
for old, new in subs:
    if old not in s:
        sys.exit("view_map3d NOT FOUND:\n" + old[:200])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("view_map3d: passes the flora through")

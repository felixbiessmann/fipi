# coding: utf-8
import os
import re
import logging
import requests
import dataset
from lxml import html
from urllib.parse import urljoin
from normdatei.text import clean_text, clean_name, fingerprint
from normdatei.parties import search_party_names

# Copied from https://github.com/bundestag/plpr-scraper/blob/master/scraper.py
# modified to support python 3 and json download

log = logging.getLogger(__name__)


DATA_PATH = os.environ.get('DATA_PATH', 'data')
TXT_DIR = os.path.join(DATA_PATH, 'txt')
OUT_DIR = os.path.join(DATA_PATH, 'out')

INDEX_URL = 'https://www.bundestag.de/plenarprotokolle'
ARCHIVE_URL = 'http://webarchiv.bundestag.de/archive/2013/0927/dokumente/protokolle/plenarprotokolle/plenarprotokolle/17%03.d.txt'

CHAIRS = [u'Vizepräsidentin', u'Vizepräsident', u'Präsident', u'Präsidentin', u'Alterspräsident', u'Alterspräsidentin']

SPEAKER_STOPWORDS = ['ich zitiere', 'zitieren', 'Zitat', 'zitiert',
                     'ich rufe den', 'ich rufe die',
                     'wir kommen zur Frage', 'kommen wir zu Frage', 'bei Frage',
                     'fordert', 'fordern', u'Ich möchte',
                     'Darin steht', ' Aspekte ', ' Punkte ', 'Berichtszeitraum']

BEGIN_MARK = re.compile('Beginn: [X\d]{1,2}.\d{1,2} Uhr')
END_MARK = re.compile('(\(Schluss:.\d{1,2}.\d{1,2}.Uhr\).*|Schluss der Sitzung)')
SPEAKER_MARK = re.compile('  (.{5,140}):\s*$')
TOP_MARK = re.compile('.*(rufe.*die Frage|zur Frage|der Tagesordnung|Tagesordnungspunkt|Zusatzpunkt).*')
POI_MARK = re.compile('\((.*)\)\s*$', re.M)
WRITING_BEGIN = re.compile('.*werden die Reden zu Protokoll genommen.*')
WRITING_END = re.compile(u'(^Tagesordnungspunkt .*:\s*$|– Drucksache d{2}/\d{2,6} –.*|^Ich schließe die Aussprache.$)')

# POI_PREFIXES = re.compile(u'(Ge ?genruf|Weiterer Zuruf|Zuruf|Weiterer)( de[sr] (Abg.|Staatsministers|Bundesministers|Parl. Staatssekretärin))?')
# REM_CHAIRS = '|'.join(CHAIRS)
# NAME_REMOVE = re.compile(u'(\\[.*\\]|\\(.*\\)|%s|^Abg.? |Liedvortrag|Bundeskanzler(in)?|, zur.*|, auf die| an die|, an .*|, Parl\\. .*|gewandt|, Staatsmin.*|, Bundesmin.*|, Ministe.*)' % REM_CHAIRS, re.U)

db = os.environ.get('DATABASE_URI', 'sqlite:///data.sqlite')
eng = dataset.connect(db)
table = eng['de_bundestag_plpr']


class SpeechParser(object):

    def __init__(self, lines):
        self.lines = lines
        self.missing_recon = False

    def parse_pois(self, group):
        for poi in group.split(' - '):
            text = poi
            speaker_name = None
            sinfo = poi.split(': ', 1)
            if len(sinfo) > 1:
                speaker_name = sinfo[0]
                text = sinfo[1]
            yield (speaker_name, text)

    def __iter__(self):
        self.in_session = False
        speaker = None
        in_writing = False
        chair_ = [False]
        text = []

        def emit(reset_chair=True):
            data = {
                'speaker': speaker,
                'in_writing': in_writing,
                'type': 'chair' if chair_[0] else 'speech',
                'text': "\n\n".join(text).strip()
            }
            if reset_chair:
                chair_[0] = False
            [text.pop() for i in xrange(len(text))]
            return data

        for line in self.lines:
            rline = line.strip()

            if not self.in_session and BEGIN_MARK.match(line):
                self.in_session = True
                continue
            elif not self.in_session:
                continue

            if END_MARK.match(rline):
                return

            if WRITING_BEGIN.match(rline):
                in_writing = True

            if WRITING_END.match(rline):
                in_writing = False

            if not len(rline):
                continue

            is_top = False
            if TOP_MARK.match(line):
                is_top = True

            has_stopword = False
            for sw in SPEAKER_STOPWORDS:
                if sw.lower() in line.lower():
                    has_stopword = True

            m = SPEAKER_MARK.match(line)
            if m is not None and not is_top and not has_stopword:
                if speaker is not None:
                    yield emit()
                _speaker = m.group(1)
                role = line.strip().split(' ')[0]
                speaker = _speaker
                chair_[0] = role in CHAIRS
                continue

            m = POI_MARK.match(rline)
            if m is not None:
                if not m.group(1).lower().strip().startswith('siehe'):
                    yield emit(reset_chair=False)
                    in_writing = False
                    for _speaker, _text in self.parse_pois(m.group(1)):
                        yield {
                            'speaker': _speaker,
                            'in_writing': False,
                            'type': 'poi',
                            'text': _text
                        }
                    continue

            text.append(rline)
        yield emit()


def file_metadata(filename):
    fname = os.path.basename(filename)
    return int(fname[:2]), int(fname[2:5])


names = set()


def parse_transcript(filename):
    wp, session = file_metadata(filename)
    with open(filename, 'rb') as fh:
        text = clean_text(fh.read())
    table.delete(wahlperiode=wp, sitzung=session)

    base_data = {
        'filename': filename,
        'sitzung': session,
        'wahlperiode': wp
    }
    print("Loading transcript: %s/%.3d, from %s" % (wp, session, filename))
    seq = 0
    parser = SpeechParser(text.split('\n'))

    for contrib in parser:
        contrib.update(base_data)
        contrib['sequence'] = seq
        contrib['speaker_cleaned'] = clean_name(contrib['speaker'])
        contrib['speaker_fp'] = fingerprint(contrib['speaker_cleaned'])
        contrib['speaker_party'] = search_party_names(contrib['speaker'])
        seq += 1
        table.insert(contrib)

    q = '''SELECT * FROM data WHERE wahlperiode = :w AND sitzung = :s
            ORDER BY sequence ASC'''
    fcsv = os.path.basename(filename).replace('.txt', '.csv')
    rp = eng.query(q, w=wp, s=session)
    dataset.freeze(rp, filename=fcsv, prefix=OUT_DIR, format='csv')


def fetch_protokolle():
    for d in TXT_DIR, OUT_DIR:
        try:
            os.makedirs(d)
        except:
            pass

    urls = set()
    res = requests.get(INDEX_URL)
    doc = html.fromstring(res.content)
    for a in doc.findall('.//a'):
        url = urljoin(INDEX_URL, a.get('href'))
        if url.endswith('.txt'):
            urls.add(url)

    for i in range(30, 260):
        url = ARCHIVE_URL % i
        urls.add(url)

    for url in urls:
        txt_file = os.path.join(TXT_DIR, os.path.basename(url))
        txt_file = txt_file.replace('-data', '')
        if os.path.exists(txt_file):
            continue

        r = requests.get(url)
        if r.status_code < 300:
            with open(txt_file, 'wb') as fh:
                fh.write(r.content)

            print(url, txt_file)

def parse_transcript_json(filename):

    wp, session = file_metadata(filename)
    with open(filename, 'rb') as fh:
        text = clean_text(fh.read())

    data = []

    base_data = {
        'filename': filename,
        'sitzung': session,
        'wahlperiode': wp
    }
    print("Loading transcript: %s/%.3d, from %s" % (wp, session, filename))
    seq = 0
    parser = SpeechParser(text.split('\n'))

    for contrib in parser:
        contrib.update(base_data)
        contrib['sequence'] = seq
        contrib['speaker_cleaned'] = clean_name(contrib['speaker'])
        contrib['speaker_fp'] = fingerprint(contrib['speaker_cleaned'])
        contrib['speaker_party'] = search_party_names(contrib['speaker'])
        seq += 1
        data.insert(0,contrib)

    jsonfile = os.path.basename(filename).replace('.txt', '.json')
    json.dump(data,open(os.path.join(OUT_DIR,jsonfile),'wb'))

if __name__ == '__main__':
    fetch_protokolle()

    for filename in os.listdir(TXT_DIR):
        parse_transcript_json(os.path.join(TXT_DIR, filename))

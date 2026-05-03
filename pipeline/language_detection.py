"""language_detection.py

Language + code-switching detection for Indian conversational data.

Two complementary layers:

1. **Script-based heuristic** (zero-dependency, always available):
   - Detects Devanagari (Hindi/Marwadi), Gurmukhi (Punjabi), Latin.
   - Token-level stats to flag code switching inside a single utterance.

2. **FastText lid.176** (optional, local model file, no API):
   - Called per segment to refine language labels when enough chars present.
   - Graceful fallback to heuristic if model file unavailable.

Segment-level detection uses Whisper TranscriptSegment start/end timestamps
to build `language_segments` with real time boundaries, enabling downstream
tools to map language to speaker turns accurately.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .transcription import TranscriptSegment

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Romanised marker sets — compiled from SemEval-2020, FIRE, ICON shared tasks,
# IIT Bombay Hinglish corpus, UD Punjabi treebank, Grierson (Rajasthani) data.
# Excludes tokens that appear commonly in English to avoid false positives.
# ---------------------------------------------------------------------------

ROMAN_HINDI_MARKERS = {
    # ── copula / auxiliaries ────────────────────────────────────────────────
    "hai", "hain", "hoon", "hun", "hona", "hoga", "hogi", "honge", "hoge",
    "hua", "hui", "hue", "ho",
    "tha", "thi", "the", "raha", "rahi", "rahe", "raho", "rehna",
    "rehta", "rehti", "rehte",
    # ── negation ────────────────────────────────────────────────────────────
    "nahi", "nahin", "nai", "nah", "mat", "maat",
    # ── question words ──────────────────────────────────────────────────────
    "kya", "kyun", "kyunki", "kyonki", "kaise", "kaisa", "kaisi",
    "kahan", "kahaan", "kab", "kaun", "kaunsa", "kaunsi",
    "kitna", "kitne", "kitni", "kisko", "kisne", "kise", "kisliye",
    "jiska", "jiski", "jiske",
    # ── affirmatives / discourse fillers ────────────────────────────────────
    "haan", "haanji", "hanji", "han",
    "acha", "accha", "achha", "theek", "thik",
    "bilkul", "zaroor", "zarur", "zaruri", "sahi", "pakka", "pakki",
    "samajh", "pata", "matlab", "mane",
    "bas", "abhi", "aji", "ab", "toh", "bhi",
    "phir", "fir", "agar", "yadi", "lekin", "kintu", "parantu",
    "magar", "isliye", "isiliye", "warna", "varna",
    "aur", "ya", "par", "pe", "se", "ko", "ka", "ke", "ki", "ne",
    "jab", "jabki", "jabse", "jaise", "waise", "vaise",
    "tab", "phir", "jab", "jadon",
    # ── pronouns ────────────────────────────────────────────────────────────
    "mai", "main", "mujhe", "mujhko", "mera", "meri", "mere",
    "tu", "tujhe",
    "tum", "tumhe", "tumhara", "tumhari", "tumhare", "tumko",
    "tera", "teri", "tere",
    "aap", "aapko", "aapka", "aapki", "aapke",
    "woh", "wo", "usse", "usko", "uska", "uski", "unhe", "unko",
    "unka", "unki", "unke", "inhone", "unhone", "inhe",
    "yeh", "ye", "isse", "isko", "iska", "iski", "iske",
    "hum", "hamara", "hamari", "hamare", "humhe", "humko",
    "apna", "apni", "apne",
    # ── address / kinship ───────────────────────────────────────────────────
    "bhai", "bhaiya", "yaar", "yaara", "ji", "sahab", "sahib",
    "didi", "behan", "behen", "beta", "beti", "baba", "amma",
    "papa", "pitaji", "mataji", "dada", "dadi", "nana", "nani",
    "chacha", "mama", "mamu", "mausi", "chachi", "tau",
    "bhabhi", "ladka", "ladki", "baccha", "bacche",
    # ── core verbs (roots + common inflections) ──────────────────────────────
    "karna", "karo", "karta", "karti", "karte", "kiya", "kiye",
    "karunga", "karungi", "karein", "kardiya", "karle", "karde", "kardenge",
    "batao", "batana", "bata", "boli", "bola", "bole", "bolo", "bolna",
    "bolega", "bolungi",
    "suno", "sunna", "suna", "suni", "sune",
    "dekho", "dekhna", "dekha", "dekhi", "dekhe",
    "aana", "aao", "aaya", "aayi", "aaye", "aaja", "aaenge",
    "jana", "jao", "jaa", "gaya", "gayi", "gaye", "jayega", "jayegi", "jaayenge",
    "lena", "liya", "liye", "lelo", "lunga",
    "dena", "diya", "diye", "dedo", "dunga",
    "padhna", "padha", "padho", "likhna", "likhna", "likho",
    "khana", "khao", "khaya", "khayi", "khaye",
    "peena", "piyo", "piya",
    "sona", "sota", "soti", "sote", "soya", "soyi", "soye",
    "uthna", "utha", "utho",
    "baithna", "baitha", "baitho",
    "rona", "rota", "roti", "rote", "roya",
    "banana", "banata", "banati", "banate", "bana", "bani", "bane", "bano",
    "chahiye", "chahte", "chahti", "chahna", "chahta", "chaha",
    "chahunga",
    "lagta", "lagti", "lagte", "laga", "lagi", "lage",
    "milta", "milti", "milte", "mila", "mili", "mile", "milega", "milegi",
    "milenge",
    "samajhna", "samajho", "samjha", "samjhao", "samjhe",
    "rehna", "raho",
    "chalna", "chalta", "chalti", "chalte", "chala", "chali", "chale",
    "chalo", "chaliye",
    "sakna", "sakta", "sakti", "sakte", "saka", "saki", "sake",
    "paana", "paata", "paati", "paate", "paya", "payi",
    "rakhna", "rakha", "rakhi", "rakhe", "rakho",
    "sochna", "sochta", "sochti", "sochte", "socha", "socho",
    "jaanna", "jaano",
    # ── common nouns ────────────────────────────────────────────────────────
    "ghar", "kaam", "log", "din", "raat", "subah", "shaam", "dopahar",
    "waqt", "samay", "aaj", "kal", "parso", "aajkal", "kabhi", "hamesha",
    "aksar",
    "paise", "paisa", "rupaye", "rupaya",
    "cheez", "baat", "khabar", "naam",
    "aadmi", "aurat", "insaan", "baccha", "dost", "shahar", "gaon",
    "zindagi", "jaan", "dil", "dimag", "dimaag", "mann",
    "ankhein", "haath", "pair", "muh", "sar", "pet",
    "rasta", "sadak", "gali", "mohalla", "baazar", "bazar", "dukaan",
    "daftar", "rishta", "rishtedaar", "shaadi", "tyohar",
    "khana", "paani", "chai", "roti", "daal", "sabzi", "meetha",
    "kitaab", "gadi", "kapda", "joota", "kagaz",
    "seva", "shikayat", "pareshani", "dikkat", "mushkil", "samasya",
    "jankari", "jaankari", "intezaar", "madat",
    # ── adjectives ──────────────────────────────────────────────────────────
    "acchi", "acche", "bura", "buri", "bure",
    "bada", "badi", "bade", "chhota", "chhoti", "chhote",
    "sundar", "khoobsurat", "aasaan", "asan",
    "saaf", "ganda", "tez", "dhima",
    "khush", "udaas", "gussa", "thaka", "bhooka", "pyaasa",
    "ameer", "garib", "sasta", "mehnga",
    "naya", "nayi", "naye", "purana", "purani", "purane",
    "poora", "poori", "poore", "seedha", "ulta", "pakka", "kaccha",
    "garm", "thanda",
    # ── adverbs / intensifiers ───────────────────────────────────────────────
    "bahut", "bohot", "bohat", "thoda", "thodi", "thode",
    "zyada", "jyada", "kaafi", "kafi", "itna", "itni", "itne",
    "jara", "zara", "sirf", "ekdum", "dono", "teeno",
    "sab", "sabhi", "har",
    "idhar", "udhar", "yahan", "wahan",
    "upar", "neeche", "baad", "pehle", "saath",
    "andar", "bahar", "paas", "dur", "tak",
    "mein", "me", "aage", "peeche",
    # ── expressions / social ─────────────────────────────────────────────────
    "arre", "arey", "arrey", "oye", "wah", "waah", "mast",
    "badhiya", "jhakkas", "jhakaas", "bindaas",
    "shukriya", "dhanyavad", "dhanyawad", "maafi", "namaste", "namaskar",
    "fikar", "tension", "chinta", "ghabrao",
    # ── wala constructions ───────────────────────────────────────────────────
    "wala", "wali", "wale", "waala", "waali", "waale",
    # ── more verbs ───────────────────────────────────────────────────────────
    "maarna", "maar", "maaro", "mara", "mari",
    "pakadna", "pakad", "pakdo", "pakda", "pakdi",
    "rokna", "roko", "roka", "roki",
    "kholna", "kholo", "khola", "kholi",
    "bandh", "bandna",
    "bhejana", "bhejo", "bheja", "bheji",
    "laana", "lao", "laya", "layi", "laaye",
    "chhodo", "chhoda", "chhodi", "chhode", "chodhna",
    "dhundna", "dhundho", "dhoondha", "dhoondhe",
    "jagao", "jaga", "jagrna",
    "hasna", "haso", "hasa", "hasi",
    "uthao", "uthana",
    "dikhao", "dikhana", "sunao",
    "chodna", "chhodna",
    "pakaana", "pakaya",
    "girna", "giro", "gira", "giri",
    "daudna", "daudo", "dauda", "daudi",
    "kheelna", "khelo", "khela", "kheli",
    # ── more nouns ───────────────────────────────────────────────────────────
    "parivaar", "naukri", "vyapaar", "dhandha",
    "mehman", "bimaar", "dawai", "dawa", "aspatal",
    "karz", "karza", "jugaad", "jugaadu",
    "desh", "gaana", "naatak", "tamasha", "khel",
    "dard", "sukoon", "khushi", "dukh", "pyaar", "mohabbat",
    "izzat", "sharm", "gussa", "nafrat",
    "sapna", "sach", "jhooth", "ummeed", "kismat",
    "zindagi", "maut", "duniya", "aasman", "zameen",
    # ── more adjectives ──────────────────────────────────────────────────────
    "taza", "basi", "teekha", "kadwa",
    "naram", "mazbut", "kamzor",
    "shayad", "lagbhag",
    "turant", "fauran", "jaldi", "dheere", "dhire",
    "ziddi", "shareef", "besharam",
    "seedhe",
    # ── Whisper ASR spelling variants ────────────────────────────────────────
    "nhi", "nhii", "pta", "kro", "bta", "smjh",
    "acha", "thk", "hna", "hn", "haa",
}

ROMAN_PUNJABI_MARKERS = {
    # ── greetings / religious ────────────────────────────────────────────────
    "waheguru", "waheguruji", "akaal", "fateh", "rabba",
    # ── pronouns (distinctly Punjabi) ───────────────────────────────────────
    "tusi", "tussi", "asi", "aapan", "aape",
    "mainu", "menu", "tenu", "tainu", "sanu", "saanu", "ohnu", "onu",
    "sannu", "tuhanu", "inna", "ohna",
    # ── address ─────────────────────────────────────────────────────────────
    "paaji", "paji", "veerji", "bhaji", "bhenji", "bhena",
    "veere", "veer", "puttar", "kuri", "munda",
    "bebe", "dadi", "nani", "tayi", "chacha", "fufad", "masi", "maasi",
    "mama", "nana",
    # ── question words (distinctly Punjabi) ─────────────────────────────────
    "kiven", "kivein", "kithe", "kitthe", "kithey",
    "kidda", "kiddan", "kado", "kadon", "kaddo",
    "kiha", "kedon", "kyon", "kon", "kinna", "kidhar",
    # ── Punjabi postpositions / particles ────────────────────────────────────
    "da", "di", "de", "nu", "ton", "wich", "vich", "utte", "ute", "heth",
    "naal", "layi", "vaaste", "kol", "agge", "pichhe", "pehlan",
    "thalle", "duwale",
    "teh", "vi", "hun", "aje", "hune", "hunne",
    "ik", "ikk", "ikko",
    # ── conjunctions / connectors ────────────────────────────────────────────
    "jad", "jado", "jadon", "tan", "taan", "jive", "jivein",
    "chahe", "bhalke", "athva", "nale", "hor", "hora",
    # ── verbs (distinctly Punjabi forms) ────────────────────────────────────
    "honda", "hundi", "hunde", "hona",
    "aunda", "aundi", "auna",
    "jaanda", "jaandi", "jauna",
    "lagda", "lagdi",
    "karda", "kardi", "karde",
    "kehnda",
    "dassna", "dasso", "dassi", "das",
    "vekhna", "vekho",
    "launa", "dinda", "dindi",
    "rakkhna", "pharna", "sochna",
    "parhna", "likhna",
    "milna", "lugna", "chalanna",
    "chhado", "chaddo",
    # ── common words / adjectives ────────────────────────────────────────────
    "changa", "changaa", "sohna", "sohni",
    "vadda", "vaddhi", "vaddhe",
    "navan", "navi", "nave",
    "pehla", "duja", "tija",
    "lamba", "lambi", "mota", "moti", "pattla",
    "kala", "kali", "chita", "chiti", "lal", "pila", "hara", "nila",
    "sukha", "geela", "thanda", "tatta", "mitta", "kaurhaa",
    "sachchi", "sach",
    # ── common nouns (Punjabi-specific or distinct) ──────────────────────────
    "pind", "kheth", "darya", "nadi", "bagh",
    "angan", "chatth", "darwaza", "khirki", "satth", "baithak", "rasoi",
    "lassi", "makhan", "sag",
    "aje", "parso",
    # ── social expressions ───────────────────────────────────────────────────
    "shukriya", "meharbani", "meherbani", "dhannvaad",
    "baut", "bhoat",
    "oye", "oi", "naah", "waah", "shukar", "shabash",
    "chalange", "chaliye",
    "sat", "sri",
    # ── more Punjabi verbs (habitual aspect -da/-di/-de) ─────────────────────
    "bolda", "boldi", "bolde",
    "sunda", "sundi", "sunde",
    "peenda", "peendi", "peende",
    "khanda", "khandi", "khande",
    "unda", "undi", "unde",
    "renda", "rendi", "rende",
    "chal",
    # ── distinctly Punjabi nouns / words ─────────────────────────────────────
    "gall", "galla",          # matter/talk (Hindi: baat)
    "wakhat",                 # time (Hindi: waqt variant)
    "udeek",                  # wait (Hindi: intezaar)
    "maada", "maadi",         # bad (Hindi: bura/buri)
    "changi",                 # good-f (Hindi: achhi)
    "bhala", "bhalaa",        # noble/good
    "chakk", "chakko",        # take! (imperative)
    "panj", "satt", "ath", "nau",  # five/seven/eight/nine
    "paratha", "parontha",    # flatbread
    "kadhi",                  # yogurt curry
    "sarson", "gur",          # mustard / jaggery
    "kheer",                  # rice pudding
    "nimbu",                  # lemon
    "reh",                    # stay (Punjabi imperative)
    "le",                     # take (Punjabi)
    "aa",                     # come (Punjabi imperative)
    "puchh", "puchho",        # ask (Punjabi)
    "rehn",                   # to stay/remain
    "aaunde", "jaande",       # coming/going (plural habitual)
    "sachchi", "pakki",       # truly/certainly (Punjabi emphasis)
}

ROMAN_MARWADI_MARKERS = {
    # ── pronouns / possessives (distinctly Marwadi) ──────────────────────────
    "mhane", "mharo", "mhari", "mhara", "mhare", "mhaari",
    "thane", "tharo", "thari", "thara", "thaara", "thaari", "thaaro",
    "aapan", "aapne", "aapro", "aapri",
    "amara", "amari", "amaro",
    "oda", "odi", "yeda", "yedi",
    "keno", "keni", "karo", "kari", "jiko", "jiki",
    # ── genitive particles (ro/ri/ra = Hindi ka/ki/ke) ───────────────────────
    "ro", "ri", "ra",
    # ── postpositions (Marwadi-specific) ─────────────────────────────────────
    "su", "syu", "soo",
    "lagi", "laagi",
    "sathe", "saathai",
    "aagai", "pachhai",
    "thalay", "thale",
    "paasey", "paase",
    # ── demonstratives ───────────────────────────────────────────────────────
    "itno", "itni", "utno", "utni",
    "ketro", "ketri", "jetro", "jetri",
    # ── question words (Marwadi-specific) ────────────────────────────────────
    "koon", "kun", "kayse",
    "kyaan", "kyan",
    "kad", "kadai", "kadey",
    "kairo", "kairi",
    "kanto", "kitro",
    # ── copula / aux (chhe = is/are, highly distinctive) ────────────────────
    "chhe", "che", "chho", "chu", "chhun", "chha",
    "hto", "hti", "hata", "hati",
    "thayo", "thayi", "huve", "huvu",
    # ── verb forms (Marwadi inflections -yo/-yi endings) ────────────────────
    "avyo", "avyi", "aavyo", "aavyi",
    "karyo", "karyi", "karyu",
    "gayio", "gayu", "gayiu",
    "aayo", "aayi",
    "jaasyu", "jaasi", "jaasyo",
    "karso", "karsi",
    "leyo", "leyi", "diyo", "diyi", "khayi",
    "jaavo", "aavo", "levo", "devo", "khavo", "revo",
    "padvano", "daudvo", "hasvo", "rovano", "mangvo",
    "milvo", "samjhvo", "jaanvo", "dekhavo", "laavo",
    "chalvo", "thakvo",
    "raheyo", "rahiyo", "basi",
    # ── greetings / address ──────────────────────────────────────────────────
    "khamma", "padharo",
    "baisa", "baaisa", "bhaisa", "bhaiji",
    "maasa", "bapusa", "babosa",
    "kaka", "kaki", "sethji",
    "sa", "saa",
    # ── quantifiers (ghano = bahut, distinctly Marwadi) ─────────────────────
    "ghano", "ghanu", "ghani", "ghana",
    "thodo", "thodi",
    # ── numbers (Marwadi variants) ────────────────────────────────────────────
    "be",       # two (Hindi: do)
    "tran",     # three (Hindi: teen)
    "nav",      # nine (Hindi: nau)
    "hajar",    # thousand (Hindi: hazaar)
    # ── distinct lexical words ───────────────────────────────────────────────
    "chhoro", "chhori",   # boy/girl (Hindi: ladka/ladki)
    "moto", "moti",       # big (Hindi: bada/badi)
    "nano", "nani",       # small (Hindi: chhota/chhoti)
    "rotlo",              # thick millet flatbread (Marwadi)
    "baati",              # baked wheat ball (Rajasthani)
    "churma",             # sweet wheat prep (Marwadi)
    "dhandho",            # business/trade (Marwadi-Gujarati)
    "lekho", "lekha",     # account ledger (merchant term)
    "hundi",              # bill of exchange (Marwari banking)
    "sahukar",            # moneylender
    "nakad",              # cash (Hindi: naqdh)
    "mol", "molbhav",     # price/bargaining
    "kuvo",               # well (Hindi: kuan)
    "talai",              # pond (Hindi: talab)
    "savero",             # morning (Hindi: subah)
    "dophar",             # afternoon
    "sanjh",              # evening
    "tadi",               # early dawn
    "sidho",              # straight (Hindi: seedha)
    "ghani",              # very (adjective fem; Hindi: bahut)
    "hun", "haaley",      # now/right now
    "kad", "kadai",       # when (Hindi: kab)
    "kyaan",              # where (Hindi: kahan)
    "kirpa",              # grace/blessing
    "aabhar",             # thank you (formal Rajasthani)
    "manas",              # person/human
    "kaado",              # mud/clay (Hindi: keechad)
    "uunt", "unt",        # camel
    "pagdi", "pagri",     # turban
    "odhni", "odhai",     # head covering
    "mojdi", "mojri",     # embroidered shoes (Rajasthani)
    "haveli",             # mansion
    "mela",               # fair/festival gathering
    "roj", "rozana",      # daily
    "hafto",              # week
    "mahino",             # month
    "baras",              # year (Hindi: saal)
    "paso",               # day before/after yesterday
    "ar", "aar",          # and (Hindi: aur — Marwadi form)
    "jad",                # when (Marwadi)
    "jyanh",              # where (Marwadi)
    "ayaan", "iythaan",   # here
    "utthaan",            # there
    "aadai", "udai",      # here/there (directional)
    "bai",                # honorific (sister/ma'am)
    "bapu",               # father
    "jai",                # victory/greeting prefix
    "padharo",            # welcome (come inside)
    "acho", "achho",      # good/okay (Marwadi form)
    "chhad",              # leave it/let go
    # ── more Marwadi verb infinitives (-vano/-no endings) ────────────────────
    "bolvano", "sunvano", "dekhvano",
    "khanvano", "pivano", "sovano",
    "uthvano", "bethvano",
    # ── more merchant / cultural vocabulary ──────────────────────────────────
    "rokda",              # cash (Marwadi-Gujarati; different from nakad)
    "khaata",             # account ledger
    "hisaab",             # calculation/account
    "vyapar",             # business (formal)
    "bopari",             # trader/merchant
    "ghewar",             # iconic Rajasthani sweet
    "malpua",             # Rajasthani sweet
    "rabdi",              # sweet milk dish
    # ── more social / conditional particles ──────────────────────────────────
    "bayaan", "bayani",   # brother/sister (Marwadi)
    "je",                 # if (Marwadi; Hindi: agar)
    "bhagwan",            # God
    "dhandha",            # business (variant)
}


# Marwadi-specific words in Devanagari — used to distinguish Marwadi from
# standard Hindi when Devanagari script is detected in the text.
# Source: Grierson "Linguistic Survey of India" Vol. IX, CIIL Marwari materials.
DEVANAGARI_MARWADI_MARKERS = {
    "म्हारो", "म्हारी", "म्हाने", "म्हारा",   # my (Hindi: मेरा/मेरी)
    "थारो", "थारी", "थाने", "थारा",             # your (Hindi: तेरा/तुम्हारा)
    "घणो", "घणी", "घणा",                         # very/much (Hindi: बहुत)
    "छोरो", "छोरी",                              # boy/girl (Hindi: लड़का/लड़की)
    "कठे", "कठै",                                # where (Hindi: कहाँ)
    "कद", "कदे",                                 # when (Hindi: कब)
    "खम्मा",                                     # greeting (unique Marwadi)
    "पधारो",                                     # welcome (Hindi: पधारिए)
    "छे", "चे",                                  # is/are copula (Hindi: है)
    "मोटो", "मोटी",                              # big (Hindi: बड़ा/बड़ी)
    "नानो", "नानी",                              # small (Hindi: छोटा/छोटी)
    "सूँ", "सूं",                                # from/with (Hindi: से)
    "आपरो", "आपरी", "आपरा",                     # our (Hindi: हमारा)
    "जको", "जकी",                                # whoever (relative pronoun)
    "घर रो", "घर री",                            # genitive constructions
    "आगाई", "पाछाई",                             # front/back
    "कुवो",                                      # well (Hindi: कुआँ)
    "छोरा", "छोरां",                             # boys/plural
    "हुवो", "हुई",                               # happened (Marwadi perfective)
    "जावणो", "आवणो",                             # to go/come (Marwadi infinitive)
    "करणो", "बोलणो",                             # to do/speak
    "थको", "थकी",                                # tired (Marwadi form)
}


@dataclass
class LanguageSegment:
    start: float
    end: float
    language: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "language": self.language,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class LanguageReport:
    primary_language: str
    confidence: float
    dominant_language: str = ""      # most common language by duration/segment count
    code_switching: bool = False
    multilingual_flag: bool = False  # True when 2+ distinct languages detected
    switching_frequency: float = 0.0 # language switches per minute of audio
    scripts: List[str] = field(default_factory=list)
    switching_score: float = 0.0     # normalized [0, 1] score for dataset filtering
    language_segments: List[LanguageSegment] = field(default_factory=list)
    # Backward-compat: per_segment kept as flat list (text + language only).
    per_segment: List[Dict] = field(default_factory=list)
    method: str = "heuristic"

    def __post_init__(self) -> None:
        if not self.dominant_language:
            self.dominant_language = self.primary_language

    def to_dict(self) -> dict:
        return {
            "primary_language": self.primary_language,
            "confidence": round(float(self.confidence), 3),
            "dominant_language": self.dominant_language,
            "code_switching": bool(self.code_switching),
            "multilingual_flag": bool(self.multilingual_flag),
            "switching_frequency": round(float(self.switching_frequency), 4),
            "switching_score": round(float(self.switching_score), 4),
            "scripts": list(self.scripts),
            "language_segments": [s.to_dict() for s in self.language_segments],
            "per_segment": self.per_segment,
            "method": self.method,
        }


def _detect_scripts(text: str) -> List[str]:
    scripts = set()
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "DEVANAGARI" in name:
            scripts.add("Devanagari")
        elif "GURMUKHI" in name:
            scripts.add("Gurmukhi")
        elif "LATIN" in name:
            scripts.add("Latin")
        elif "BENGALI" in name:
            scripts.add("Bengali")
        elif "TAMIL" in name:
            scripts.add("Tamil")
        elif "TELUGU" in name:
            scripts.add("Telugu")
    return sorted(scripts)


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[\w']+", text.lower())


def _probe_marwadi_devanagari(text: str) -> float:
    """Return confidence [0, 1] that Devanagari text is Marwadi, not Hindi.

    Checks for Marwadi-specific Devanagari tokens: distinctive pronouns
    (म्हारो/थारो), copula (छे), quantifier (घणो), etc.  Even 1-2 hits in a
    short segment is strong evidence because these forms don't appear in
    standard Hindi.
    """
    if not text.strip():
        return 0.0
    # Tokenise on whitespace + Devanagari danda (।) and common punctuation.
    tokens = set(re.split(r"[\s।॥,।.!?]+", text))
    hits = sum(1 for m in DEVANAGARI_MARWADI_MARKERS if m in tokens or m in text)
    if hits == 0:
        return 0.0
    return min(1.0, hits / max(len(tokens), 1) * 6)


def _roman_indic_probe(tokens: List[str]) -> Tuple[str, float]:
    """Return (language_code, strength) if enough Indic markers found.

    Tie-breaking: Marwadi > Punjabi > Hindi.  Hindi is the default ASR fallback
    so specificity is rewarded; we only override when evidence is clear.
    """
    if not tokens:
        return ("en", 0.0)
    hits_hi = sum(1 for t in tokens if t in ROMAN_HINDI_MARKERS)
    hits_pa = sum(1 for t in tokens if t in ROMAN_PUNJABI_MARKERS)
    hits_mw = sum(1 for t in tokens if t in ROMAN_MARWADI_MARKERS)

    total = len(tokens)
    # Secondary sort key (0/1/2) breaks ties toward more specific languages.
    best = max(
        (hits_hi, 0, "hi-Latn"),
        (hits_pa, 1, "pa-Latn"),
        (hits_mw, 2, "mwr-Latn"),
        key=lambda x: (x[0], x[1]),
    )
    hits, _, lang = best
    ratio = hits / max(total, 1)
    if hits >= 2 and ratio >= 0.05:
        return (lang, min(1.0, ratio * 4))
    return ("en", 0.0)


def _classify_segment_language(
    text: str,
    fasttext_lid: "FastTextLID | None",
    roman_indic_classifier,
) -> Tuple[str, float]:
    """Return (language_code, confidence) for a single segment text."""
    seg_scripts = _detect_scripts(text)
    seg_tokens = _tokenise(text)

    if "Devanagari" in seg_scripts:
        mw_conf = _probe_marwadi_devanagari(text)
        if mw_conf >= 0.25:
            return ("mwr", max(0.70, mw_conf))
        return ("hi", 0.9)
    if "Gurmukhi" in seg_scripts:
        return ("pa", 0.9)

    # Latin-only: try ML classifier → lexicon → fasttext → fallback
    seg_lang, seg_conf = "en", 0.6
    ml_used = False

    if roman_indic_classifier is not None and roman_indic_classifier.available():
        pred = roman_indic_classifier.predict(text)
        if pred is not None and pred.confidence >= 0.55:
            seg_lang, seg_conf = pred.language, pred.confidence
            ml_used = True

    if not ml_used:
        probed_lang, strength = _roman_indic_probe(seg_tokens)
        if probed_lang != "en" and strength > 0.1:
            seg_lang = probed_lang
            seg_conf = 0.55 + 0.3 * strength

    if fasttext_lid and fasttext_lid.available() and text.strip() and not ml_used:
        ft_lang, ft_conf = fasttext_lid.predict(text)
        if ft_lang != "und":
            probed_lang, strength = _roman_indic_probe(seg_tokens)
            if probed_lang != "en" and strength > 0.2:
                seg_lang = probed_lang
            else:
                seg_lang, seg_conf = ft_lang, ft_conf

    return seg_lang, seg_conf


class FastTextLID:
    """Wraps the lid.176 fasttext model if the user has downloaded it."""

    DEFAULT_PATH = os.environ.get(
        "FASTTEXT_LID_MODEL",
        os.path.expanduser("~/.cache/sonexis/lid.176.ftz"),
    )

    def __init__(self, path: Optional[str] = None):
        self.path = path or self.DEFAULT_PATH
        self._model = None

    def available(self) -> bool:
        return os.path.isfile(self.path)

    def _load(self):
        if self._model is None:
            import fasttext
            self._model = fasttext.load_model(self.path)
        return self._model

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.available() or not text.strip():
            return ("und", 0.0)
        try:
            model = self._load()
            labels, probs = model.predict(text.replace("\n", " "), k=1)
            lang = labels[0].replace("__label__", "") if labels else "und"
            return (lang, float(probs[0]) if len(probs) else 0.0)
        except Exception as err:
            log.warning("fasttext predict failed: %s", err)
            return ("und", 0.0)


def _compute_switching_frequency(
    language_segments: List[LanguageSegment],
    total_duration_s: float,
) -> float:
    """Return language switches per minute."""
    if len(language_segments) < 2 or total_duration_s <= 0:
        return 0.0
    switches = sum(
        1 for a, b in zip(language_segments, language_segments[1:])
        if a.language != b.language
    )
    return round(switches / (total_duration_s / 60.0), 4)


def detect_language(
    full_text: str,
    segments_text: Optional[List[str]] = None,
    fasttext_lid: Optional[FastTextLID] = None,
    roman_indic_classifier=None,
    # New: pass TranscriptSegment list directly for timestamp-aware detection.
    transcript_segments: "Optional[List[TranscriptSegment]]" = None,
    total_duration_s: float = 0.0,
) -> LanguageReport:
    """Return a combined language + code-switching report.

    When transcript_segments is provided (preferred), language_segments will
    have real start/end timestamps. Otherwise falls back to segments_text
    (text-only, no timestamps).

    Resolution order for Latin-only text:
      1. roman_indic_classifier (trained ML model) if supplied.
      2. Romanised-Indic lexicon probe.
      3. FastText lid.176 if available.
      4. Fall back to "en" at 0.6 confidence.
    """
    segments_text = segments_text or []
    scripts = _detect_scripts(full_text)
    tokens = _tokenise(full_text)

    # Script-based global baseline.
    if "Devanagari" in scripts and "Latin" not in scripts:
        mw_conf = _probe_marwadi_devanagari(full_text)
        if mw_conf >= 0.25:
            primary, conf = "mwr", max(0.70, mw_conf)
        else:
            primary, conf = "hi", 0.9
    elif "Gurmukhi" in scripts and "Latin" not in scripts:
        primary, conf = "pa", 0.9
    elif "Devanagari" in scripts and "Latin" in scripts:
        mw_conf = _probe_marwadi_devanagari(full_text)
        if mw_conf >= 0.25:
            primary, conf = "mwr", max(0.65, mw_conf)
        else:
            primary, conf = "hi", 0.75
    elif "Gurmukhi" in scripts and "Latin" in scripts:
        primary, conf = "pa", 0.75
    else:
        primary, conf = "en", 0.6
        ml_used = False
        if roman_indic_classifier is not None and roman_indic_classifier.available():
            pred = roman_indic_classifier.predict(full_text)
            if pred is not None and pred.confidence >= 0.55:
                primary, conf = pred.language, pred.confidence
                ml_used = True
        if not ml_used:
            probed_lang, probe_strength = _roman_indic_probe(tokens)
            if probed_lang != "en":
                primary, conf = probed_lang, 0.55 + 0.3 * probe_strength

    method = "heuristic"
    if roman_indic_classifier is not None and roman_indic_classifier.available():
        method = "ml+heuristic"

    if fasttext_lid and fasttext_lid.available() and full_text.strip():
        ft_lang, ft_conf = fasttext_lid.predict(full_text)
        method = "ml+fasttext+heuristic" if "ml" in method else "fasttext+heuristic"
        if ft_lang != "und":
            if scripts == ["Devanagari"]:
                primary, conf = ("hi", max(conf, ft_conf))
            elif scripts == ["Gurmukhi"]:
                primary, conf = ("pa", max(conf, ft_conf))
            elif scripts == ["Latin"] and "ml" not in method:
                probed_lang, probe_strength = _roman_indic_probe(tokens)
                if probed_lang != "en" and probe_strength > 0.2:
                    primary = probed_lang
                else:
                    primary, conf = ft_lang, ft_conf

    # ------------------------------------------------------------------ #
    #  Per-segment language detection with timestamps.
    # ------------------------------------------------------------------ #
    language_segments: List[LanguageSegment] = []
    per_segment: List[Dict] = []
    seen_langs: Counter = Counter()

    if transcript_segments:
        for seg in transcript_segments:
            if not seg.text.strip():
                continue
            seg_lang, seg_conf = _classify_segment_language(
                seg.text, fasttext_lid, roman_indic_classifier
            )
            language_segments.append(LanguageSegment(
                start=seg.start,
                end=seg.end,
                language=seg_lang,
                confidence=seg_conf,
            ))
            per_segment.append({"text": seg.text, "language": seg_lang})
            seg_dur = max(0.0, seg.end - seg.start)
            seen_langs[seg_lang] += seg_dur if seg_dur > 0 else 1
    else:
        # Fallback: text-only (no timestamps available).
        for text in segments_text:
            if not text.strip():
                continue
            seg_lang, seg_conf = _classify_segment_language(
                text, fasttext_lid, roman_indic_classifier
            )
            per_segment.append({"text": text, "language": seg_lang})
            seen_langs[seg_lang] += 1

    # Dominant language = highest duration/count.
    dominant_language = seen_langs.most_common(1)[0][0] if seen_langs else primary

    # Use transcript_segments total duration if not supplied.
    dur = total_duration_s
    if dur <= 0 and transcript_segments:
        dur = max((s.end for s in transcript_segments), default=0.0)

    switching_freq = _compute_switching_frequency(language_segments, dur)
    switching_score = float(min(1.0, switching_freq / 10.0))
    multilingual_flag = len(seen_langs) >= 2
    code_switching = multilingual_flag or len(scripts) >= 2

    return LanguageReport(
        primary_language=primary,
        confidence=float(conf),
        dominant_language=dominant_language,
        code_switching=bool(code_switching),
        multilingual_flag=bool(multilingual_flag),
        switching_frequency=switching_freq,
        switching_score=switching_score,
        scripts=scripts,
        language_segments=language_segments,
        per_segment=per_segment,
        method=method,
    )


def detect_language_per_speaker(
    turns: "List",  # List[SpeakerTurn]
    transcript_segments: "Optional[List[TranscriptSegment]]",
    fasttext_lid: Optional[FastTextLID] = None,
    roman_indic_classifier=None,
) -> Dict[str, LanguageReport]:
    """Detect language for each speaker independently.

    Maps transcript segments to speakers by finding the turn that best
    overlaps each segment, then builds a per-speaker LanguageReport.

    Returns {speaker_id: LanguageReport}.
    """
    if not turns or not transcript_segments:
        return {}

    # Build a quick list of (start, end, speaker) from turns for overlap lookup.
    turn_list = sorted(turns, key=lambda t: t.start)

    def _best_speaker_for_segment(seg_start: float, seg_end: float) -> str:
        """Return the speaker whose turn most overlaps this segment."""
        best_spk = turn_list[0].speaker
        best_overlap = 0.0
        for turn in turn_list:
            if turn.start > seg_end:
                break
            overlap = min(turn.end, seg_end) - max(turn.start, seg_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = turn.speaker
        return best_spk

    # Partition transcript segments by speaker.
    spk_segs: Dict[str, list] = {}
    for seg in transcript_segments:
        if not seg.text.strip():
            continue
        spk = _best_speaker_for_segment(seg.start, seg.end)
        spk_segs.setdefault(spk, []).append(seg)

    reports: Dict[str, LanguageReport] = {}
    for spk, segs in spk_segs.items():
        full_text = " ".join(s.text for s in segs)
        total_dur = max((s.end for s in segs), default=0.0)
        reports[spk] = detect_language(
            full_text=full_text,
            fasttext_lid=fasttext_lid,
            roman_indic_classifier=roman_indic_classifier,
            transcript_segments=segs,
            total_duration_s=total_dur,
        )

    return reports

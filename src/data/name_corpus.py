"""Multilingual name corpus with romanization variant families.

Every entry pairs a native-script form with the set of romanizations that a
real feed would carry for it. This is what makes the generated data an actual
test of the matcher: the watchlist may hold "Muhammad", the wire message may
carry "Mohamed", and the customer record may hold "محمد", and all three have to
resolve to the same entity without the generator ever telling the matcher so.

Variant sets are drawn from the conventions that actually collide in practice:
  - Arabic: vowel-insertion ambiguity (the abjad writes no short vowels), and
    the q/k and dh/z consonant merges of regional romanization standards.
  - Cyrillic: BGN/PCGN vs. ISO 9 vs. legacy Anglicised endings (-ov/-off/-ow),
    plus the gendered surname suffix that makes Ivanov and Ivanova the same
    family and different people.
  - Han: Hanyu Pinyin vs. Wade-Giles vs. Cantonese Jyutping, which is why Zhang,
    Chang and Cheung are one surname across three passports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class NameForm:
    native: str
    variants: tuple[str, ...]

    @property
    def canonical(self) -> str:
        return self.variants[0]


def _f(native: str, *variants: str) -> NameForm:
    return NameForm(native=native, variants=variants)


_CURATED_ARABIC_GIVEN: Final[tuple[NameForm, ...]] = (
    _f("محمد", "Muhammad", "Mohammed", "Mohamed", "Muhammed", "Mohammad"),
    _f("أحمد", "Ahmad", "Ahmed", "Ahmet"),
    _f("علي", "Ali", "Aly", "Alie"),
    _f("حسن", "Hasan", "Hassan", "Hassen"),
    _f("حسين", "Husayn", "Hussein", "Hussain", "Husain"),
    _f("عبدالله", "Abdullah", "Abdallah", "Abdulla", "Abd Allah"),
    _f("عبدالرحمن", "Abdulrahman", "Abdul Rahman", "Abdurrahman"),
    _f("عبدالعزيز", "Abdulaziz", "Abdul Aziz", "Abdelaziz"),
    _f("خالد", "Khalid", "Khaled", "Halid"),
    _f("عمر", "Umar", "Omar", "Omer"),
    _f("عثمان", "Uthman", "Othman", "Osman"),
    _f("إبراهيم", "Ibrahim", "Ibraheem", "Brahim"),
    _f("يوسف", "Yusuf", "Yousef", "Yousuf", "Youssef"),
    _f("سعيد", "Said", "Saeed", "Sayeed"),
    _f("سلمان", "Salman", "Selman"),
    _f("فيصل", "Faysal", "Faisal", "Feisal"),
    _f("ناصر", "Nasir", "Nasser", "Naser"),
    _f("طارق", "Tariq", "Tarek", "Tarik"),
    _f("رشيد", "Rashid", "Rasheed", "Rachid"),
    _f("كريم", "Karim", "Kareem", "Carim"),
    _f("مصطفى", "Mustafa", "Moustafa", "Mustapha"),
    _f("محمود", "Mahmud", "Mahmoud", "Mahmood"),
    _f("جمال", "Jamal", "Gamal", "Djamal"),
    _f("سامي", "Sami", "Samy", "Samih"),
    _f("وليد", "Walid", "Waleed", "Oualid"),
    _f("ياسر", "Yasir", "Yasser", "Yaser"),
    _f("زياد", "Ziyad", "Ziad", "Zayd"),
    _f("هشام", "Hisham", "Hicham", "Hesham"),
    _f("بشار", "Bashar", "Bachar", "Beshar"),
    _f("قاسم", "Qasim", "Kassim", "Kasim", "Qassem"),
    _f("صالح", "Salih", "Saleh", "Salah"),
    _f("عادل", "Adil", "Adel", "Aadil"),
    _f("نبيل", "Nabil", "Nabeel"),
    _f("فريد", "Farid", "Fareed", "Ferid"),
    _f("رامي", "Rami", "Ramy"),
    _f("ماهر", "Mahir", "Maher"),
    _f("أنور", "Anwar", "Anouar", "Enver"),
    _f("سليم", "Salim", "Saleem", "Selim"),
    _f("أمين", "Amin", "Ameen", "Emin"),
    _f("عثمان", "Uthman", "Osman", "Othmane"),
    _f("يعقوب", "Yaqub", "Yacoub", "Yakub"),
    _f("إسماعيل", "Ismail", "Ismael", "Esmail"),
    _f("سليمان", "Sulayman", "Suleiman", "Soliman"),
    _f("هارون", "Harun", "Haroun", "Aaron"),
    _f("فاطمة", "Fatima", "Fatma", "Fatimah"),
    _f("عائشة", "Aisha", "Ayesha", "Aicha"),
    _f("زينب", "Zaynab", "Zeinab", "Zainab"),
    _f("مريم", "Maryam", "Mariam", "Marium"),
    _f("نور", "Nur", "Noor", "Nour"),
    _f("ليلى", "Layla", "Leila", "Laila"),
)

_CURATED_ARABIC_FAMILY: Final[tuple[NameForm, ...]] = (
    _f("الحربي", "Harbi", "Al-Harbi", "Alharbi"),
    _f("القحطاني", "Qahtani", "Al-Qahtani", "Kahtani"),
    _f("العتيبي", "Utaybi", "Al-Otaibi", "Alutaibi"),
    _f("الغامدي", "Ghamdi", "Al-Ghamdi", "Alghamdi"),
    _f("الزهراني", "Zahrani", "Al-Zahrani", "Alzahrani"),
    _f("الشهري", "Shahri", "Al-Shehri", "Alshahri"),
    _f("البغدادي", "Baghdadi", "Al-Baghdadi", "Bagdadi"),
    _f("الدمشقي", "Dimashqi", "Al-Dimashqi", "Dimashki"),
    _f("المصري", "Misri", "Al-Masri", "Elmasri"),
    _f("العراقي", "Iraqi", "Al-Iraqi", "Iraki"),
    _f("الجزائري", "Jazairi", "Al-Jazairi", "Djazairi"),
    _f("اللبناني", "Lubnani", "Al-Lubnani", "Loubnani"),
    _f("التكريتي", "Tikriti", "Al-Tikriti", "Takriti"),
    _f("الحسيني", "Husayni", "Al-Husseini", "Hosseini"),
    _f("الشامي", "Shami", "Al-Shami", "Chami"),
    _f("الكردي", "Kurdi", "Al-Kurdi", "Kourdi"),
    _f("النجار", "Najjar", "Al-Najjar", "Nagar"),
    _f("الخطيب", "Khatib", "Al-Khatib", "Elkhatib"),
    _f("الحاج", "Hajj", "El-Hajj", "Haj"),
    _f("الصباح", "Sabah", "Al-Sabah", "Sabbah"),
    _f("الرشيد", "Rashid", "Al-Rashid", "Alrasheed"),
    _f("الفهد", "Fahd", "Al-Fahd", "Alfahad"),
    _f("السعيد", "Saeed", "Al-Saeed", "Alsaid"),
    _f("الحمد", "Hamad", "Al-Hamad", "Alhamd"),
    _f("العلي", "Ali", "Al-Ali", "Alaly"),
    _f("الجابر", "Jabir", "Al-Jaber", "Elgaber"),
    _f("المالكي", "Maliki", "Al-Maliki", "Malki"),
    _f("الشمري", "Shammari", "Al-Shammari", "Chammari"),
    _f("الدوسري", "Dosari", "Al-Dosari", "Dossary"),
    _f("العنزي", "Anzi", "Al-Anzi", "Anazi"),
)

# Cyrillic surnames carry the -a feminine suffix. The masculine and feminine
# forms are the same family and different individuals, which is the single most
# common false-positive pattern on Russia-program screening.
_CURATED_CYRILLIC_GIVEN: Final[tuple[NameForm, ...]] = (
    _f("Владимир", "Vladimir", "Wladimir", "Volodymyr"),
    _f("Александр", "Aleksandr", "Alexander", "Alexandr", "Oleksandr"),
    _f("Сергей", "Sergei", "Sergey", "Serguei", "Serhii"),
    _f("Дмитрий", "Dmitry", "Dmitri", "Dmitriy"),
    _f("Андрей", "Andrei", "Andrey", "Andriy"),
    _f("Алексей", "Aleksei", "Alexey", "Alexei"),
    _f("Николай", "Nikolai", "Nikolay", "Mykola"),
    _f("Иван", "Ivan", "Ivane", "Iwan"),
    _f("Михаил", "Mikhail", "Michail", "Mykhailo"),
    _f("Юрий", "Yuri", "Yury", "Iurii"),
    _f("Игорь", "Igor", "Ihor", "Igore"),
    _f("Виктор", "Viktor", "Victor"),
    _f("Павел", "Pavel", "Pavlo", "Pawel"),
    _f("Олег", "Oleg", "Oleh"),
    _f("Роман", "Roman", "Romane"),
    _f("Максим", "Maksim", "Maxim", "Maksym"),
    _f("Евгений", "Evgeny", "Yevgeny", "Yevhen"),
    _f("Константин", "Konstantin", "Constantin", "Kostiantyn"),
    _f("Артем", "Artem", "Artyom", "Artiom"),
    _f("Денис", "Denis", "Denys"),
    _f("Анатолий", "Anatoly", "Anatoli", "Anatolii"),
    _f("Борис", "Boris", "Borys"),
    _f("Григорий", "Grigory", "Grigori", "Hryhorii"),
    _f("Леонид", "Leonid", "Leonyd"),
    _f("Валерий", "Valery", "Valeri", "Valerii"),
    _f("Екатерина", "Ekaterina", "Yekaterina", "Kateryna"),
    _f("Наталья", "Natalya", "Natalia", "Nataliya"),
    _f("Ольга", "Olga", "Olha"),
    _f("Татьяна", "Tatyana", "Tatiana", "Tetiana"),
    _f("Елена", "Elena", "Yelena", "Olena"),
)

_CURATED_CYRILLIC_FAMILY: Final[tuple[NameForm, ...]] = (
    _f("Иванов", "Ivanov", "Ivanoff", "Iwanow"),
    _f("Смирнов", "Smirnov", "Smirnoff", "Smyrnov"),
    _f("Кузнецов", "Kuznetsov", "Kuznetzov", "Kouznetsov"),
    _f("Попов", "Popov", "Popoff", "Popow"),
    _f("Соколов", "Sokolov", "Sokoloff", "Sokolow"),
    _f("Лебедев", "Lebedev", "Lebedeff", "Lebediev"),
    _f("Козлов", "Kozlov", "Kozloff", "Koslow"),
    _f("Новиков", "Novikov", "Novikoff", "Nowikow"),
    _f("Морозов", "Morozov", "Morozoff", "Morosow"),
    _f("Петров", "Petrov", "Petroff", "Petrow"),
    _f("Волков", "Volkov", "Volkoff", "Wolkow"),
    _f("Соловьев", "Solovyev", "Soloviev", "Solovieff"),
    _f("Васильев", "Vasilyev", "Vasiliev", "Wassiliew"),
    _f("Зайцев", "Zaytsev", "Zaitsev", "Zaitzeff"),
    _f("Павлов", "Pavlov", "Pavloff", "Pawlow"),
    _f("Семенов", "Semenov", "Semyonov", "Semenoff"),
    _f("Голубев", "Golubev", "Golubeff", "Holubiev"),
    _f("Виноградов", "Vinogradov", "Winogradow", "Vinogradoff"),
    _f("Богданов", "Bogdanov", "Bogdanoff", "Bohdanov"),
    _f("Медведев", "Medvedev", "Medvedeff", "Medwedew"),
    _f("Тимошенко", "Tymoshenko", "Timoshenko", "Tymochenko"),
    _f("Шевченко", "Shevchenko", "Chevtchenko", "Schewtschenko"),
    _f("Коваленко", "Kovalenko", "Kowalenko", "Covalenco"),
    _f("Бондаренко", "Bondarenko", "Bondarenco"),
    _f("Лукашенко", "Lukashenko", "Loukachenko", "Lukaschenka"),
    _f("Абрамович", "Abramovich", "Abramovitch", "Abramowitsch"),
    _f("Дерипаска", "Deripaska", "Deripasca"),
    _f("Ротенберг", "Rotenberg", "Rothenberg"),
    _f("Тимченко", "Timchenko", "Tymchenko"),
    _f("Сечин", "Sechin", "Setchin", "Sietschin"),
)

_CURATED_HAN_SURNAME: Final[tuple[NameForm, ...]] = (
    _f("王", "Wang", "Wong", "Ong"),
    _f("李", "Li", "Lee", "Ly"),
    _f("张", "Zhang", "Chang", "Cheung"),
    _f("刘", "Liu", "Lau", "Lew"),
    _f("陈", "Chen", "Chan", "Tan"),
    _f("杨", "Yang", "Yeung", "Young"),
    _f("黄", "Huang", "Wong", "Hwang"),
    _f("赵", "Zhao", "Chao", "Chiu"),
    _f("吴", "Wu", "Ng", "Goh"),
    _f("周", "Zhou", "Chow", "Chou"),
    _f("徐", "Xu", "Hsu", "Tsui"),
    _f("孙", "Sun", "Suen", "Soon"),
    _f("马", "Ma", "Mah", "Mar"),
    _f("朱", "Zhu", "Chu", "Choo"),
    _f("胡", "Hu", "Wu", "Woo"),
    _f("郭", "Guo", "Kuo", "Kwok"),
    _f("何", "He", "Ho", "Hoh"),
    _f("高", "Gao", "Kao", "Ko"),
    _f("林", "Lin", "Lam", "Lim"),
    _f("罗", "Luo", "Lo", "Law"),
    _f("郑", "Zheng", "Cheng", "Chung"),
    _f("梁", "Liang", "Leung", "Neo"),
    _f("谢", "Xie", "Hsieh", "Tse"),
    _f("宋", "Song", "Sung", "Soong"),
    _f("唐", "Tang", "Tong", "Thong"),
    _f("韩", "Han", "Hon", "Hahn"),
    _f("冯", "Feng", "Fung", "Fong"),
    _f("邓", "Deng", "Teng", "Tang"),
    _f("曹", "Cao", "Tsao", "Cho"),
    _f("彭", "Peng", "Pang", "Phang"),
)

_CURATED_HAN_GIVEN: Final[tuple[NameForm, ...]] = (
    _f("伟", "Wei", "Wai"),
    _f("芳", "Fang", "Fong"),
    _f("娜", "Na", "Nah"),
    _f("敏", "Min", "Man"),
    _f("静", "Jing", "Ching"),
    _f("强", "Qiang", "Chiang"),
    _f("磊", "Lei", "Louie"),
    _f("洋", "Yang", "Yeung"),
    _f("勇", "Yong", "Yung"),
    _f("军", "Jun", "Chun"),
    _f("杰", "Jie", "Kit"),
    _f("涛", "Tao", "Tou"),
    _f("明", "Ming", "Meng"),
    _f("超", "Chao", "Chiu"),
    _f("秀英", "Xiuying", "Sau Ying"),
    _f("建国", "Jianguo", "Kin Kwok"),
    _f("志强", "Zhiqiang", "Chi Keung"),
    _f("小明", "Xiaoming", "Siu Ming"),
    _f("国庆", "Guoqing", "Kwok Hing"),
    _f("海燕", "Haiyan", "Hoi Yin"),
    _f("建华", "Jianhua", "Kin Wah"),
    _f("桂英", "Guiying", "Kwai Ying"),
    _f("文", "Wen", "Man"),
    _f("斌", "Bin", "Ban"),
    _f("鹏", "Peng", "Pang"),
)

_CURATED_LATIN_GIVEN: Final[tuple[NameForm, ...]] = tuple(
    _f(n, n)
    for n in (
        "James", "John", "Robert", "Michael", "William", "David", "Richard",
        "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
        "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret",
        "Carlos", "Juan", "Luis", "Jose", "Miguel", "Rafael", "Eduardo",
        "Ana", "Maria", "Carmen", "Lucia", "Sofia", "Isabel",
        "Hans", "Klaus", "Stefan", "Jurgen", "Wolfgang", "Dieter",
        "Pierre", "Jean", "Michel", "Philippe", "Laurent", "Olivier",
    )
)

_CURATED_LATIN_FAMILY: Final[tuple[NameForm, ...]] = tuple(
    _f(n, n)
    for n in (
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
        "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Muller", "Schmidt", "Schneider", "Fischer",
        "Weber", "Meyer", "Wagner", "Becker", "Hoffmann", "Dubois", "Bernard",
        "Petit", "Durand", "Leroy", "Moreau", "Rossi", "Ferrari", "Esposito",
        "Bianchi", "Romano", "Colombo", "Silva", "Santos", "Ferreira",
        "Kowalski", "Nowak", "Wojcik", "Kaminski", "Novak", "Horvath",
    )
)

ORG_PREFIX: Final[tuple[str, ...]] = (
    "Global", "United", "Pioneer", "Meridian", "Crescent", "Falcon", "Delta",
    "Anchor", "Summit", "Orion", "Vanguard", "Pacific", "Atlantic", "Northern",
    "Southern", "Sterling", "Cedar", "Granite", "Horizon", "Aurora", "Zenith",
    "Emerald", "Sapphire", "Titan", "Apex", "Nexus", "Prime", "Regal",
)

ORG_CORE: Final[tuple[str, ...]] = (
    "Trading", "Shipping", "Petroleum", "Logistics", "Metals", "Chemicals",
    "Maritime", "Commodities", "Holdings", "Engineering", "Aviation",
    "Electronics", "Textiles", "Construction", "Investments", "Minerals",
    "Machinery", "Agro", "Marine", "Energy", "Freight", "Industries",
)

ORG_SUFFIX: Final[tuple[str, ...]] = (
    "LLC", "Ltd", "Limited", "Inc", "Corp", "GmbH", "SA", "SARL", "BV", "PLC",
    "Pte Ltd", "Co", "AG", "OOO", "JSC", "PJSC", "FZE", "DMCC",
)

HONORIFICS: Final[tuple[str, ...]] = (
    "Mr.", "Dr.", "Sheikh", "Hajji", "Sayyid", "Gen.", "Col.", "Eng.",
    "Prof.", "Mullah", "Capt.", "Sr.",
)

# Occupations feed the customer profile and, indirectly, the segment baselines
# that the peer-relative monitoring scenarios compare against.
OCCUPATIONS: Final[tuple[str, ...]] = (
    "Salaried Employee", "Retail Merchant", "Restaurant Owner", "Import Export",
    "Construction Contractor", "Freight Forwarder", "Money Service Business",
    "Consultant", "Physician", "Teacher", "Software Engineer", "Retired",
    "Student", "Car Dealer", "Jeweller", "Property Developer", "Farmer",
    "Logistics Manager", "Commodity Trader", "Precious Metals Dealer",
)


# ---------------------------------------------------------------------------
# Long tail
# ---------------------------------------------------------------------------
# The curated lists above carry the variant families the matcher is tested on,
# but 5,000 watchlist entries drawn from ~120 surnames would give the blocking
# index unrealistically dense buckets: every surname block would hold dozens of
# entries and pruning would look far worse than it is in production, where the
# SDN list spans thousands of distinct family names.
#
# The tail below is built morphologically rather than hand-listed, because that
# is how these surnames are actually formed: a Slavic surname is a root plus a
# possessive suffix, an Arabic family name is frequently a nisba (place or tribe
# of origin plus -i), and both carry predictable romanization alternates.

_CYRILLIC_ROOTS: Final[tuple[tuple[str, str], ...]] = (
    ("Бел", "Bel"), ("Черн", "Chern"), ("Красн", "Krasn"), ("Зелен", "Zelen"),
    ("Гор", "Gor"), ("Дуб", "Dub"), ("Соснов", "Sosnov"), ("Берез", "Berez"),
    ("Волч", "Volch"), ("Медвед", "Medved"), ("Лис", "Lis"), ("Зайц", "Zayts"),
    ("Орл", "Orl"), ("Сокол", "Sokol"), ("Ястреб", "Yastreb"), ("Голуб", "Golub"),
    ("Воробь", "Vorob"), ("Журавл", "Zhuravl"), ("Рыб", "Ryb"), ("Щук", "Shchuk"),
    ("Карп", "Karp"), ("Гончар", "Gonchar"), ("Ткач", "Tkach"), ("Портн", "Portn"),
    ("Мельник", "Melnik"), ("Пекар", "Pekar"), ("Плотник", "Plotnik"),
    ("Столяр", "Stolyar"), ("Кожевник", "Kozhevnik"), ("Бондар", "Bondar"),
    ("Коваль", "Koval"), ("Стрельц", "Strelts"), ("Пушкар", "Pushkar"),
    ("Казак", "Kazak"), ("Атаман", "Ataman"), ("Хмель", "Khmel"),
    ("Овчар", "Ovchar"), ("Пастух", "Pastukh"), ("Табак", "Tabak"),
    ("Виноград", "Vinograd"), ("Малин", "Malin"), ("Рябин", "Ryabin"),
    ("Топол", "Topol"), ("Кедр", "Kedr"), ("Ясен", "Yasen"),
)

_CYRILLIC_SUFFIXES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("ов", ("ov", "off", "ow")),
    ("ев", ("ev", "eff", "ew")),
    ("ин", ("in", "ine", "yn")),
    ("ский", ("sky", "ski", "skiy")),
    ("енко", ("enko", "enco")),
)

_ARABIC_NISBA_ROOTS: Final[tuple[tuple[str, str], ...]] = (
    ("حلب", "Halab"), ("حمص", "Hims"), ("حماة", "Hama"), ("درعا", "Daraa"),
    ("ادلب", "Idlib"), ("الرقة", "Raqqa"), ("موصل", "Mawsil"), ("بصرة", "Basra"),
    ("نجف", "Najaf"), ("كربلاء", "Karbala"), ("رمادي", "Ramadi"),
    ("فلوجة", "Falluja"), ("سامراء", "Samarra"), ("كركوك", "Kirkuk"),
    ("اربيل", "Arbil"), ("عمان", "Amman"), ("زرقاء", "Zarqa"), ("اربد", "Irbid"),
    ("صنعاء", "Sana"), ("عدن", "Adan"), ("تعز", "Taiz"), ("حضرموت", "Hadramawt"),
    ("مأرب", "Marib"), ("طرابلس", "Tarabulus"), ("بنغازي", "Banghazi"),
    ("مصراتة", "Misrata"), ("سبها", "Sabha"), ("درنة", "Darna"), ("تونس", "Tunis"),
    ("صفاقس", "Safaqis"), ("قابس", "Qabis"), ("وهران", "Wahran"),
    ("قسنطينة", "Qusantina"), ("عنابة", "Annaba"), ("فاس", "Fas"),
    ("مكناس", "Miknas"), ("طنجة", "Tanja"), ("جدة", "Jidda"), ("مكة", "Makka"),
    ("مدينة", "Madina"), ("ابها", "Abha"), ("تبوك", "Tabuk"), ("جيزان", "Jizan"),
    ("قطيف", "Qatif"), ("هفوف", "Hufuf"),
)

_HAN_TAIL_SURNAMES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("郑", ("Zheng", "Cheng", "Chung")), ("梁", ("Liang", "Leung", "Neo")),
    ("谢", ("Xie", "Hsieh", "Tse")), ("宋", ("Song", "Sung", "Soong")),
    ("韩", ("Han", "Hon", "Hahn")), ("冯", ("Feng", "Fung", "Fong")),
    ("曾", ("Zeng", "Tseng", "Tsang")), ("肖", ("Xiao", "Hsiao", "Siu")),
    ("田", ("Tian", "Tien", "Chan")), ("董", ("Dong", "Tung", "Tong")),
    ("袁", ("Yuan", "Yuen", "Yen")), ("潘", ("Pan", "Poon", "Phua")),
    ("蒋", ("Jiang", "Chiang", "Cheung")), ("蔡", ("Cai", "Tsai", "Choy")),
    ("杜", ("Du", "Tu", "To")), ("叶", ("Ye", "Yeh", "Ip")),
    ("程", ("Cheng", "Ching", "Chin")), ("苏", ("Su", "Soo", "So")),
    ("魏", ("Wei", "Ngai", "Wai")), ("丁", ("Ding", "Ting", "Teng")),
    ("沈", ("Shen", "Shum", "Sim")), ("姚", ("Yao", "Yiu", "Yew")),
    ("卢", ("Lu", "Lo", "Loo")), ("崔", ("Cui", "Tsui", "Choi")),
    ("钟", ("Zhong", "Chung", "Chong")), ("谭", ("Tan", "Tam", "Ham")),
    ("汪", ("Wang", "Wong", "Ong")), ("范", ("Fan", "Fam", "Huan")),
    ("金", ("Jin", "Kam", "Kim")), ("石", ("Shi", "Shek", "Sek")),
    ("贾", ("Jia", "Chia", "Ka")), ("夏", ("Xia", "Hsia", "Ha")),
    ("方", ("Fang", "Fong", "Pong")), ("白", ("Bai", "Pak", "Pai")),
    ("孟", ("Meng", "Mang", "Mung")), ("秦", ("Qin", "Chin", "Chun")),
    ("江", ("Jiang", "Kong", "Kang")), ("薛", ("Xue", "Sit", "Hsueh")),
    ("段", ("Duan", "Tuan", "Tyun")), ("雷", ("Lei", "Louie", "Lui")),
)

_LATIN_TAIL_FAMILY: Final[tuple[str, ...]] = (
    "Bennett", "Coleman", "Delgado", "Ellison", "Fletcher", "Gallagher",
    "Hartley", "Ingram", "Jennings", "Kirkland", "Langford", "Maddox",
    "Norwood", "Ogden", "Pemberton", "Quinlan", "Radcliffe", "Sinclair",
    "Thorne", "Underwood", "Vaughan", "Whitaker", "Yates", "Ziegler",
    "Alvarez", "Beltran", "Cardenas", "Duarte", "Escobar", "Fuentes",
    "Guerrero", "Herrera", "Ibarra", "Jimenez", "Lozano", "Montoya",
    "Navarro", "Ochoa", "Pacheco", "Quintero", "Reyes", "Salazar",
    "Trevino", "Urbina", "Valdez", "Zamora", "Andersen", "Bakker",
    "Christensen", "Dijkstra", "Eriksen", "Fabbri", "Grasso", "Hofmann",
    "Iversen", "Jansen", "Klein", "Lindqvist", "Moretti", "Nilsson",
    "Olsen", "Pedersen", "Rasmussen", "Sorensen", "Thomsen", "Vermeulen",
    "Wagner", "Zimmermann", "Adeyemi", "Balogun", "Chukwu", "Diallo",
    "Eze", "Fofana", "Gueye", "Hassane", "Ibrahim", "Keita", "Lawal",
    "Mensah", "Ndiaye", "Okafor", "Sankara", "Toure", "Yeboah",
)

_LATIN_TAIL_GIVEN: Final[tuple[str, ...]] = (
    "Adrian", "Bernard", "Cedric", "Damian", "Edmund", "Felix", "Gareth",
    "Hector", "Ignacio", "Julian", "Konrad", "Leonard", "Marcus", "Nolan",
    "Oscar", "Percival", "Quentin", "Rupert", "Sebastian", "Tobias",
    "Ulrich", "Vincent", "Wesley", "Xavier", "Yannick", "Zachary",
    "Adriana", "Beatriz", "Celeste", "Delphine", "Eleanor", "Francesca",
    "Giselle", "Helena", "Ingrid", "Juliette", "Katarina", "Leonora",
    "Marisol", "Nadine", "Odette", "Priscilla", "Rosalind", "Serafina",
    "Theodora", "Ursula", "Valentina", "Wilhelmina", "Yasmine", "Zelda",
)


def _build_cyrillic_tail() -> tuple[NameForm, ...]:
    forms: list[NameForm] = []
    for native_root, latin_root in _CYRILLIC_ROOTS:
        for native_suffix, latin_suffixes in _CYRILLIC_SUFFIXES:
            forms.append(
                NameForm(
                    native=f"{native_root}{native_suffix}",
                    variants=tuple(f"{latin_root}{s}" for s in latin_suffixes),
                )
            )
    return tuple(forms)


def _build_arabic_tail() -> tuple[NameForm, ...]:
    forms: list[NameForm] = []
    for native_root, latin_root in _ARABIC_NISBA_ROOTS:
        # q/k is the single most common romanization fork in Arabic family
        # names; -i / -y is the second. Both are generated so the matcher has to
        # bridge them rather than getting one canonical spelling everywhere.
        base = f"{latin_root}i"
        variants = [base, f"Al-{base}", base.replace("q", "k").replace("Q", "K")]
        if base.endswith("i"):
            variants.append(f"{base[:-1]}y")
        stem = native_root[2:] if native_root.startswith("ال") else native_root
        forms.append(
            NameForm(
                native=f"ال{stem}ي",
                variants=tuple(dict.fromkeys(variants)),
            )
        )
    return tuple(forms)


def _build_han_tail() -> tuple[NameForm, ...]:
    return tuple(NameForm(native=native, variants=variants) for native, variants in _HAN_TAIL_SURNAMES)


CYRILLIC_FAMILY: Final[tuple[NameForm, ...]] = _CURATED_CYRILLIC_FAMILY + _build_cyrillic_tail()
ARABIC_FAMILY: Final[tuple[NameForm, ...]] = _CURATED_ARABIC_FAMILY + _build_arabic_tail()
HAN_SURNAME: Final[tuple[NameForm, ...]] = _CURATED_HAN_SURNAME + _build_han_tail()
LATIN_FAMILY: Final[tuple[NameForm, ...]] = _CURATED_LATIN_FAMILY + tuple(
    _f(n, n) for n in _LATIN_TAIL_FAMILY
)
LATIN_GIVEN: Final[tuple[NameForm, ...]] = _CURATED_LATIN_GIVEN + tuple(
    _f(n, n) for n in _LATIN_TAIL_GIVEN
)


# Given-name tails. Surname variety alone is not enough: screening collisions
# are driven by the (given, family) pair, so a 50-name given pool against a
# 5,000-entry watchlist manufactures exact-name collisions at a rate no real
# portfolio exhibits, and every one of them becomes an unearned alert.
_ARABIC_GIVEN_TAIL: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("بلال", ("Bilal", "Belal")), ("جابر", ("Jabir", "Jaber")),
    ("راشد", ("Rashed", "Rached")), ("زكريا", ("Zakaria", "Zakariya")),
    ("سعد", ("Saad", "Sad")), ("شادي", ("Shadi", "Chadi")),
    ("صابر", ("Sabir", "Saber")), ("ضياء", ("Diya", "Zia")),
    ("طلال", ("Talal", "Tallal")), ("ظافر", ("Zafir", "Dhafer")),
    ("عصام", ("Issam", "Essam")), ("غسان", ("Ghassan", "Gassan")),
    ("فادي", ("Fadi", "Fady")), ("قيس", ("Qais", "Kais")),
    ("لؤي", ("Luay", "Loai")), ("مازن", ("Mazin", "Mazen")),
    ("نادر", ("Nadir", "Nader")), ("هيثم", ("Haytham", "Haitham")),
    ("وسام", ("Wisam", "Wissam")), ("يحيى", ("Yahya", "Yehia")),
    ("أيمن", ("Ayman", "Aiman")), ("باسل", ("Basil", "Bassel")),
    ("تامر", ("Tamir", "Tamer")), ("ثامر", ("Thamir", "Thamer")),
    ("جواد", ("Jawad", "Gawad")), ("حازم", ("Hazim", "Hazem")),
    ("خضر", ("Khadir", "Khader")), ("دياب", ("Diyab", "Diab")),
    ("ذاكر", ("Dhakir", "Zaker")), ("رائد", ("Raid", "Raed")),
    ("زهير", ("Zuhayr", "Zoheir")), ("سراج", ("Siraj", "Serag")),
    ("شريف", ("Sharif", "Cherif")), ("صفوان", ("Safwan", "Safouan")),
    ("عماد", ("Imad", "Emad")), ("فهد", ("Fahd", "Fahad")),
    ("قصي", ("Qusay", "Kusai")), ("لطفي", ("Lutfi", "Lotfi")),
    ("منير", ("Munir", "Mounir")), ("نزار", ("Nizar", "Nezar")),
    ("هاني", ("Hani", "Hany")), ("وائل", ("Wail", "Wael")),
    ("يامن", ("Yamin", "Yamen")), ("أسامة", ("Usama", "Osama")),
    ("بدر", ("Badr", "Bader")), ("جهاد", ("Jihad", "Gihad")),
    ("حمزة", ("Hamza", "Hamzah")), ("خليل", ("Khalil", "Khaleel")),
    ("رياض", ("Riyad", "Riad")), ("سهيل", ("Suhayl", "Soheil")),
    ("شاكر", ("Shakir", "Chaker")), ("صادق", ("Sadiq", "Sadek")),
    ("عاصم", ("Asim", "Assem")), ("فتحي", ("Fathi", "Fathy")),
    ("كمال", ("Kamal", "Kemal")), ("مروان", ("Marwan", "Merwan")),
    ("نضال", ("Nidal", "Nedal")), ("هلال", ("Hilal", "Helal")),
    ("وضاح", ("Waddah", "Wadah")), ("يزيد", ("Yazid", "Yazeed")),
    ("سميرة", ("Samira", "Sameera")), ("هدى", ("Huda", "Hoda")),
    ("رانيا", ("Rania", "Raniya")), ("دعاء", ("Duaa", "Doaa")),
    ("سناء", ("Sanaa", "Sana")), ("امل", ("Amal", "Amel")),
    ("وفاء", ("Wafa", "Wafaa")), ("نجلاء", ("Najla", "Naglaa")),
    ("عبير", ("Abir", "Abeer")), ("منى", ("Muna", "Mona")),
)

_CYRILLIC_GIVEN_TAIL: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("Аркадий", ("Arkady", "Arkadi")), ("Богдан", ("Bogdan", "Bohdan")),
    ("Вадим", ("Vadim", "Wadim")), ("Геннадий", ("Gennady", "Hennadii")),
    ("Данил", ("Danil", "Danylo")), ("Егор", ("Egor", "Yegor")),
    ("Захар", ("Zakhar", "Zachar")), ("Илья", ("Ilya", "Illia")),
    ("Кирилл", ("Kirill", "Kyrylo")), ("Лев", ("Lev", "Leo")),
    ("Марк", ("Mark", "Marko")), ("Назар", ("Nazar", "Nazariy")),
    ("Остап", ("Ostap", "Ostapp")), ("Петр", ("Petr", "Pyotr", "Petro")),
    ("Руслан", ("Ruslan", "Rouslan")), ("Степан", ("Stepan", "Stefan")),
    ("Тарас", ("Taras", "Tarass")), ("Ульян", ("Ulyan", "Uliyan")),
    ("Федор", ("Fedor", "Fyodor")), ("Харитон", ("Khariton", "Hariton")),
    ("Цезарь", ("Tsezar", "Cezar")), ("Эдуард", ("Eduard", "Edward")),
    ("Юлиан", ("Yulian", "Julian")), ("Яков", ("Yakov", "Iakov")),
    ("Антон", ("Anton", "Antone")), ("Василий", ("Vasily", "Vasili")),
    ("Глеб", ("Gleb", "Hlib")), ("Дамир", ("Damir", "Damyr")),
    ("Ефим", ("Efim", "Yefim")), ("Жанна", ("Zhanna", "Janna")),
    ("Зоя", ("Zoya", "Zoia")), ("Инна", ("Inna", "Ina")),
    ("Ксения", ("Ksenia", "Kseniya")), ("Лариса", ("Larisa", "Larissa")),
    ("Марина", ("Marina", "Maryna")), ("Нина", ("Nina", "Nyna")),
    ("Оксана", ("Oksana", "Oxana")), ("Полина", ("Polina", "Pauline")),
    ("Раиса", ("Raisa", "Rayisa")), ("Светлана", ("Svetlana", "Svitlana")),
    ("Ульяна", ("Ulyana", "Uliana")), ("Фаина", ("Faina", "Fayina")),
    ("Юлия", ("Yulia", "Julia")), ("Ярослав", ("Yaroslav", "Iaroslav")),
    ("Валентин", ("Valentin", "Walentin")), ("Геворг", ("Gevorg", "Gevork")),
    ("Давид", ("David", "Davyd")), ("Ермолай", ("Ermolai", "Yermolay")),
    ("Зиновий", ("Zinovy", "Zinoviy")), ("Иосиф", ("Iosif", "Yosyp")),
    ("Климент", ("Kliment", "Klyment")), ("Лука", ("Luka", "Louka")),
    ("Матвей", ("Matvei", "Matviy")), ("Никита", ("Nikita", "Mykyta")),
    ("Олесь", ("Oles", "Olesh")), ("Прохор", ("Prokhor", "Prohor")),
    ("Родион", ("Rodion", "Rodyon")), ("Савва", ("Savva", "Sava")),
    ("Тимур", ("Timur", "Tymur")), ("Устин", ("Ustin", "Ustyn")),
)

_HAN_GIVEN_TAIL: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("宇轩", ("Yuxuan", "Yu Hin")), ("子豪", ("Zihao", "Tsz Ho")),
    ("浩然", ("Haoran", "Ho Yin")), ("俊杰", ("Junjie", "Chun Kit")),
    ("志明", ("Zhiming", "Chi Ming")), ("嘉豪", ("Jiahao", "Ka Ho")),
    ("家豪", ("Jiahao", "Ka Hou")), ("伟强", ("Weiqiang", "Wai Keung")),
    ("永强", ("Yongqiang", "Wing Keung")), ("国强", ("Guoqiang", "Kwok Keung")),
    ("文杰", ("Wenjie", "Man Kit")), ("俊豪", ("Junhao", "Chun Ho")),
    ("晓东", ("Xiaodong", "Hiu Tung")), ("春霞", ("Chunxia", "Chun Ha")),
    ("丽娟", ("Lijuan", "Lai Kuen")), ("秀兰", ("Xiulan", "Sau Lan")),
    ("淑芬", ("Shufen", "Suk Fan")), ("玉兰", ("Yulan", "Yuk Lan")),
    ("美玲", ("Meiling", "Mei Ling")), ("雅婷", ("Yating", "Nga Ting")),
    ("欣怡", ("Xinyi", "Yan Yi")), ("诗涵", ("Shihan", "Sze Ham")),
    ("梓涵", ("Zihan", "Tsz Ham")), ("志伟", ("Zhiwei", "Chi Wai")),
    ("建平", ("Jianping", "Kin Ping")), ("学军", ("Xuejun", "Hok Kwan")),
    ("凤英", ("Fengying", "Fung Ying")), ("兰英", ("Lanying", "Lan Ying")),
    ("红梅", ("Hongmei", "Hung Mui")), ("玉梅", ("Yumei", "Yuk Mui")),
    ("志高", ("Zhigao", "Chi Ko")), ("永生", ("Yongsheng", "Wing Sang")),
    ("立新", ("Lixin", "Lap San")), ("向东", ("Xiangdong", "Heung Tung")),
    ("卫东", ("Weidong", "Wai Tung")), ("爱国", ("Aiguo", "Oi Kwok")),
    ("振华", ("Zhenhua", "Chun Wah")), ("小龙", ("Xiaolong", "Siu Lung")),
    ("大伟", ("Dawei", "Tai Wai")), ("天明", ("Tianming", "Tin Ming")),
    ("宏伟", ("Hongwei", "Wang Wai")), ("俊民", ("Junmin", "Chun Man")),
    ("春林", ("Chunlin", "Chun Lam")), ("金龙", ("Jinlong", "Kam Lung")),
    ("永康", ("Yongkang", "Wing Hong")), ("兆明", ("Zhaoming", "Siu Ming")),
    ("耀祖", ("Yaozu", "Yiu Cho")), ("庆丰", ("Qingfeng", "Hing Fung")),
    ("德华", ("Dehua", "Tak Wah")), ("学文", ("Xuewen", "Hok Man")),
)


def _tail_forms(rows: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[NameForm, ...]:
    return tuple(NameForm(native=native, variants=variants) for native, variants in rows)


ARABIC_GIVEN: Final[tuple[NameForm, ...]] = _CURATED_ARABIC_GIVEN + _tail_forms(_ARABIC_GIVEN_TAIL)
CYRILLIC_GIVEN: Final[tuple[NameForm, ...]] = _CURATED_CYRILLIC_GIVEN + _tail_forms(
    _CYRILLIC_GIVEN_TAIL
)
HAN_GIVEN: Final[tuple[NameForm, ...]] = _CURATED_HAN_GIVEN + _tail_forms(_HAN_GIVEN_TAIL)

CORPUS_SIZES: Final[dict[str, int]] = {
    "arabic_given": len(ARABIC_GIVEN),
    "arabic_family": len(ARABIC_FAMILY),
    "cyrillic_given": len(CYRILLIC_GIVEN),
    "cyrillic_family": len(CYRILLIC_FAMILY),
    "han_given": len(HAN_GIVEN),
    "han_surname": len(HAN_SURNAME),
    "latin_given": len(LATIN_GIVEN),
    "latin_family": len(LATIN_FAMILY),
}

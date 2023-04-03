import re


class Alignment:
    def __init__(self, phones, starts, ends,accents=None) -> None:
        self.phones = phones
        self.starts = starts
        self.ends = ends
        self.accents = accents

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index+=1
        if self.index > self.phones:
            raise StopIteration
        if self.accents == None:
            return self.phones[self.index], self.starts[self.index], self.ends[self.index]
        else:
            return self.phones[self.index], self.starts[self.index], self.ends[self.index],self.accents[self.index]
    def isAccentProvided(self):
        return self.accents != None

# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))
def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    accent = []
    N = len(labels)

    for n in range(len(labels)):
        lab_curr = labels[n]


        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == 'sil':
            assert n== 0 or n == N-1
            if n == N-1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("")
                elif e3 == 1:
                    PP.append("")
            continue
        elif p3 == "pau":
            PP.append("sp")
            accent.append('0')
            continue
        else:
            PP.append(p3)
        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        lab_next = labels[n + 1]
        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", lab_next)
        # アクセント境界
        if a3 == 1 and a2_next == 1:
            accent.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            accent.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            accent.append("[")
        else:
            accent.append('0')
    return PP, accent
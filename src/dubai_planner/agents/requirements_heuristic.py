"""Rule-based request extraction: the mock provider's reply for the Requirements agent.

A deterministic, regex-based baseline so the whole pipeline runs offline. It covers
common phrasings (``10-14 Feb 2027``, ``5 days in March``, ``AED 8,000``, ``$2k``,
``family of 4``, ``kids aged 6 and 9``, ``vegetarian``, ``wheelchair``). It is **not** a
substitute for a real model (AC2.3 measures that); it gives the mock sensible output.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from dubai_planner.agents.requirements import RequirementsExtraction
from dubai_planner.models.request import AccessibilityNeed, BudgetScope, DietaryFlag, Pace
from dubai_planner.tools.search import INTEREST_SYNONYMS

MONTHS = {m: i for i, names in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
     ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
     ("oct", "october"), ("nov", "november"), ("dec", "december")], start=1) for m in names}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|" \
         r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DASH = r"\s*(?:-|–|—|to|until|till)\s*"
_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(k)?"
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "a": 1}
_INTEREST_EXTRAS = ("museums", "mosques", "souks", "theme parks", "water parks", "safari",
                    "nightlife", "photography", "malls", "dhow")
AREA_WORDS = {"marina": "Dubai Marina", "downtown": "Downtown", "jbr": "JBR", "deira": "Deira",
              "bur dubai": "Bur Dubai", "palm": "Palm Jumeirah", "jumeirah": "Jumeirah",
              "business bay": "Business Bay", "al barsha": "Al Barsha", "old dubai": "Bur Dubai",
              "creek": "Deira"}


def _year(raw: str | None, month: int, today: date) -> int:
    if raw:
        return int(raw)
    return today.year if month >= today.month else today.year + 1


def _amount(raw: str, k: str | None) -> Decimal | None:
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None
    return value * 1000 if k else value


def _num(word: str) -> int | None:
    return int(word) if word.isdigit() else _WORD_NUMBERS.get(word.lower())


def _dates(text: str, today: date) -> tuple[date | None, date | None, int | None, int | None]:
    """(start, end, month-only, year-only)."""
    iso = re.findall(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        days = [date(int(y), int(m), int(d)) for y, m, d in iso]
        return days[0], days[1] if len(days) > 1 else None, None, None
    patterns = [  # (regex, (d1, m1, d2, m2, year) group indexes; None = reuse)
        (rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}{_DASH}(\d{{1,2}})(?:st|nd|rd|th)?\s+"
         rf"{_MONTH},?\s*(\d{{4}})?", (1, 2, 3, 4, 5)),
        (rf"(\d{{1,2}})(?:st|nd|rd|th)?{_DASH}(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH},?"
         rf"\s*(\d{{4}})?", (1, 3, 2, None, 4)),
        (rf"{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?{_DASH}(\d{{1,2}})(?:st|nd|rd|th)?,?"
         rf"\s*(\d{{4}})?", (2, 1, 3, None, 4)),
    ]
    for pattern, (d1, m1, d2, m2, yr) in patterns:
        match = re.search(pattern, text)
        if match:
            g = match.group
            month1 = MONTHS[g(m1)[:3] if g(m1)[:3] in MONTHS else g(m1)]
            month2 = MONTHS[g(m2)[:3]] if m2 else month1
            year = _year(g(yr), month1, today)
            start = date(year, month1, int(g(d1)))
            end = date(year + (1 if month2 < month1 else 0), month2, int(g(d2)))
            return start, end, None, None
    single = re.search(rf"(?:from|on|starting|arriving)\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?"
                       rf"\s+(?:of\s+)?{_MONTH},?\s*(\d{{4}})?", text)
    if single:
        month = MONTHS[single.group(2)[:3]]
        return date(_year(single.group(3), month, today), month, int(single.group(1))), \
            None, None, None
    month_only = re.search(rf"\b(?:in|during|this|next|for)\s+{_MONTH}\b,?\s*(\d{{4}})?", text)
    if month_only:
        stated_year = int(month_only.group(2)) if month_only.group(2) else None
        return None, None, MONTHS[month_only.group(1)[:3]], stated_year
    return None, None, None, None


def _duration(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]*"
                      r"(day|night)s?\b", text)
    if match:
        n = _num(match.group(1))
        if n is not None:
            return n + 1 if match.group(2) == "night" else n
    if re.search(r"\b(a|one)\s+week\b", text):
        return 7
    if re.search(r"\btwo\s+weeks\b|\bfortnight\b", text):
        return 14
    if re.search(r"\bweekend\b", text):
        return 2
    return None


def _budget(text: str) -> tuple[Decimal | None, str, BudgetScope | None]:
    currency_patterns = [
        (rf"\b(?:aed|dhs?|dirhams?)\s*{_NUM}", "AED", False),
        (rf"{_NUM}\s*(?:aed|dhs?|dirhams?)\b", "AED", False),
        (rf"\$\s*{_NUM}", "USD", False),
        (rf"{_NUM}\s*(?:usd|us dollars|dollars)\b", "USD", False),
        (rf"(?:usd)\s*{_NUM}", "USD", False),
        (rf"€\s*{_NUM}|{_NUM}\s*(?:eur|euros?)\b", "EUR", True),
        (rf"£\s*{_NUM}|{_NUM}\s*(?:gbp|pounds?)\b", "GBP", True),
        (rf"(?:budget(?:\s+of|\s+is|\s+around|\s+about)?|spend)\s*(?:of\s*)?{_NUM}", "AED", False),
    ]
    for pattern, currency, two_groups in currency_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        groups = [g for g in match.groups()]
        if two_groups:
            raw, k = (groups[0], groups[1]) if groups[0] else (groups[2], groups[3])
        else:
            raw, k = groups[0], groups[1]
        amount = _amount(raw, k)
        if amount is None or amount <= 0:
            continue
        tail = text[match.end(): match.end() + 30]
        scope = None
        if re.search(r"per\s+(person|head|pax)|each|pp\b", tail):
            scope = BudgetScope.PER_PERSON
        elif re.search(r"per\s+day|a day|daily|/day", tail):
            scope = BudgetScope.PER_DAY
        return amount, currency, scope
    return None, "AED", None


def _travellers(text: str) -> tuple[int | None, int | None, list[int]]:
    ages: list[int] = []
    for match in re.finditer(r"aged?\s+((?:\d{1,2}(?:\s*,\s*|\s+and\s+|\s*&\s*)?)+)", text):
        ages += [int(a) for a in re.findall(r"\d{1,2}", match.group(1)) if int(a) <= 17]
    ages += [int(a) for a in re.findall(r"(\d{1,2})[\s-]*(?:year|yr)s?[\s-]*old", text)
             if int(a) <= 17]
    ages = ages[:10]
    adults = children = None
    if m := re.search(r"family of (\d{1,2}|three|four|five|six|seven)", text):
        total = _num(m.group(1)) or 0
        adults, children = 2, max(total - 2, 0)
    if m := re.search(r"(\d{1,2}|one|two|three|four|five|six)\s+adults?", text):
        adults = _num(m.group(1))
    if m := re.search(r"(\d{1,2}|one|two|three|four|five)\s+(?:kids|children|child|sons?|"
                      r"daughters?)", text):
        children = _num(m.group(1))
    elif children is None and ages:
        children = len(ages)
    elif children is None and re.search(r"\b(?:our|my|a|with)\s+(?:kid|child|son|daughter)\b",
                                        text):
        children = 1
    if adults is None:
        if m := re.search(r"(\d{1,2}|two|three|four|five|six)\s+(?:people|persons|travellers|"
                          r"travelers|friends|of us)", text):
            n = _num(m.group(1)) or 1
            adults = n + 1 if "friends" in m.group(0) and "with" in text else n
        elif re.search(r"\b(couple|honeymoon|my (?:wife|husband|partner|girlfriend|boyfriend)|"
                       r"me and|and i\b|we two|two of us)", text):
            adults = 2
        elif re.search(r"\b(solo|alone|by myself|just me|i am travelling|i'm travelling)\b", text):
            adults = 1
        elif children:
            adults = 2
    return adults, children, ages


def _interests(text: str) -> list[str]:
    found = [k for k in INTEREST_SYNONYMS if re.search(rf"\b{re.escape(k)}\b", text)]
    found += [k for k in _INTEREST_EXTRAS if k in text]
    return list(dict.fromkeys(found))


def heuristic_extract(raw: str, *, today: date) -> RequirementsExtraction:
    text = raw.lower()
    start, end, month, year = _dates(text, today)
    amount, currency, scope = _budget(text)
    adults, children, ages = _travellers(text)
    dietary = [f for f, pat in ((DietaryFlag.VEGAN, r"\bvegan"),
                                (DietaryFlag.VEGETARIAN, r"\bvegetarian|\bveggie|\bno meat"),
                                (DietaryFlag.HALAL, r"\bhalal"),
                                (DietaryFlag.GLUTEN_FREE, r"gluten|coeliac|celiac"))
               if re.search(pat, text)]
    access = [a for a, pat in ((AccessibilityNeed.WHEELCHAIR, r"wheelchair"),
                               (AccessibilityNeed.LIMITED_MOBILITY,
                                r"limited mobility|can't walk much|cannot walk far|elderly"),
                               (AccessibilityNeed.STROLLER, r"stroller|pram|buggy"))
              if re.search(pat, text)]
    pace = (Pace.RELAXED if re.search(r"relax|slow|easy|laid[- ]back|chill", text)
            else Pace.PACKED if re.search(r"packed|busy|as much as possible|action", text)
            else None)
    must = [m.strip() for m in re.findall(
        r"(?:must (?:see|visit|do)|definitely (?:see|visit)|want to (?:see|visit))\s+(?:the\s+)?"
        r"([a-z][a-z' ]{2,40}?)(?=[,.;!]| and |$)", text)]
    avoid = [m.strip() for m in re.findall(
        r"\b(?:avoid|no|not interested in|skip|hate)\s+([a-z][a-z ]{2,25}?)(?=[,.;!]| and |$)",
        text) if m.strip() not in {"meat", "pork", "budget", "kids", "children", "rush"}]
    areas = [name for word, name in AREA_WORDS.items() if re.search(rf"\b{word}\b", text)]
    return RequirementsExtraction(
        start_date=start, end_date=end, duration_days=_duration(text), month=month, year=year,
        adults=adults, children=children, children_ages=ages, budget_amount=amount,
        budget_currency=currency, budget_scope=scope, interests=_interests(text), pace=pace,
        dietary=dietary, accessibility=access, preferred_areas=list(dict.fromkeys(areas)),
        must_include=must, avoid=avoid,
    )

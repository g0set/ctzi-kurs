
from __future__ import annotations
import ipaddress


def inn_valid(s: str) -> bool:
    if not s.isascii() or not s.isdigit() or len(s) not in (10, 12):
        return False
    a = list(map(int, s))
    if len(a) == 10:
        k = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return sum(x*y for x, y in zip(a, k)) % 11 % 10 == a[9]
    k11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    k12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    return (sum(x*y for x, y in zip(a, k11)) % 11 % 10 == a[10]
            and sum(x*y for x, y in zip(a, k12)) % 11 % 10 == a[11])


def ogrnip_valid(s: str) -> bool:
    return (s.isascii() and s.isdigit() and len(s) == 15
            and int(s[-1]) == (int(s[:-1]) % 13) % 10)


def luhn_valid(s: str) -> bool:
    if not s.isascii() or not s.isdigit() or not (13 <= len(s) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(s)):
        n = int(ch)
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def ip_valid(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        return False

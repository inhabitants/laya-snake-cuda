"""Screen text for the disturb mode. English by default, `--lang pt` for Portuguese.

The upstream panel labels (NEXT MOVE, DEAD-END RISK, ...) stay in English in both.
"""

STRINGS = {
    "en": {
        "you": "YOU {arrow}",
        "you_tag": "YOU",
        "dodged": "DODGED",
        "cycle": "FIXED ROUTE",
        "free": "FREE ROUTE",
        "driving": "you're disturbing it",
        "recovering": "recovering...",
        "recovered": "recovered in {n} moves",
        "fell": "fell ({reason})",
        "counters": "disturbed {a:03d}  recovered {b:03d}  fell {c:03d}",
        "keys": "WASD/ARROWS disturb   +/- speed",
        "keys2": "SPACE pause   R new round   Q quit",
        "reasons": {"wall": "wall", "body": "body", "reverse": "reverse"},
    },
    "pt": {
        "you": "VOCÊ {arrow}",
        "you_tag": "VOCÊ",
        "dodged": "DESVIOU",
        "cycle": "ROTA FIXA",
        "free": "ROTA LIVRE",
        "driving": "você atrapalhando",
        "recovering": "se ajeitando...",
        "recovered": "se ajeitou em {n} jogadas",
        "fell": "caiu ({reason})",
        "counters": "atrapalhou {a:03d}  ajeitou {b:03d}  caiu {c:03d}",
        "keys": "WASD/SETAS atrapalha   +/- velocidade",
        "keys2": "ESPAÇO pausa   R nova rodada   Q sai",
        "reasons": {"wall": "parede", "body": "corpo", "reverse": "ré"},
    },
}


def text(lang):
    return STRINGS.get(lang, STRINGS["en"])

"""Короткая презентация для первой встречи: 6 слайдов.

Запуск:  python docs/build_presentation_short.py
Выход:   docs/presentation_short.pptx
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from build_presentation import (DARK, FIRE, GREY, LIGHT, OK, WATER, WHITE, blank,
                                bullets, footer, header, rect, table, text)

OUT = Path(__file__).resolve().parent / "presentation_short.pptx"
W, H = Inches(13.333), Inches(7.5)


def p(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    return s


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 — титул
    s = p(prs)
    rect(s, 0, 0, W, H, fill=DARK)
    rect(s, 0, Inches(4.78), W, Inches(0.06), fill=WATER)
    text(s, "Инженерные кадры России · ИКаР-БАС · 2026/27",
         Inches(0.9), Inches(0.85), Inches(11), Inches(0.4), size=14,
         color=RGBColor(0x9F, 0xC5, 0xD8))
    text(s, "Самолёт с бортовым ИИ\nищет пожары и людей на воде",
         Inches(0.9), Inches(1.45), Inches(11.9), Inches(1.9), size=34, bold=True,
         color=WHITE, spacing=1.1)
    text(s, "Лесные пожары  ·  Поиск людей на воде  ·  Работает без интернета",
         Inches(0.9), Inches(3.3), Inches(11.4), Inches(0.5), size=15,
         color=RGBColor(0x9F, 0xC5, 0xD8))
    text(s, "Команда «Guslitron774» · ДТ «Кванториум», Челябинск\n"
            "Партнёр: Главное управление МЧС России по Челябинской области",
         Inches(0.9), Inches(3.8), Inches(11.4), Inches(0.9), size=16, color=WHITE)
    for i, chip in enumerate(["Пожары", "Люди на воде", "ИИ на борту"]):
        x = Inches(0.9) + Inches(3.2) * i
        r = rect(s, x, Inches(5.35), Inches(2.95), Inches(0.62), fill=None,
                 line=RGBColor(0x4A, 0x6B, 0x86))
        text(s, chip, x, Inches(5.49), Inches(2.95), Inches(0.4), size=14,
             color=RGBColor(0xD6, 0xE4, 0xEE), align=PP_ALIGN.CENTER)

    # 2 — проблема
    s = p(prs)
    header(s, "проблема", "Патрулирование не успевает за пожаром")
    cards = [
        ("Часы на маршрут", "Десятки тысяч гектаров. Дым замечают, когда очаг "
                            "уже набрал силу.", FIRE),
        ("Коптер — 20–30 минут", "Малый дрон не покрывает большой район: "
                                 "время в воздухе — главный ограничитель.", FIRE),
        ("Человек в воде — 20 пикселей", "С берега его видно плохо, а счёт идёт "
                                         "на минуты. Люди ищут глазами.", WATER),
    ]
    for i, (t, d, col) in enumerate(cards):
        x = Inches(0.7) + Inches(4.15) * i
        rect(s, x, Inches(2.1), Inches(3.85), Inches(2.9), fill=LIGHT)
        rect(s, x, Inches(2.1), Inches(0.08), Inches(2.9), fill=col)
        text(s, t, x + Inches(0.3), Inches(2.4), Inches(3.3), Inches(0.8),
             size=19, bold=True, color=DARK)
        text(s, d, x + Inches(0.3), Inches(3.2), Inches(3.3), Inches(1.6),
             size=14, color=GREY, spacing=1.15)
    text(s, "Спутник — редкие снимки с крупным пикселем. Наблюдатель — "
            "зависит от погоды и внимания. Нужен непрерывный обзор с воздуха.",
         Inches(0.7), Inches(5.4), Inches(12), Inches(0.8), size=16, color=DARK)
    footer(s, 2)

    # 3 — концепт
    s = p(prs)
    header(s, "концепт", "Самолёт даёт время, ИИ — непрерывное внимание")
    boxes = [
        ("Носитель", "Самолётный БПЛА", "Часы в воздухе вместо десятков минут."),
        ("Интеллект", "Две модели", "«Лес»: дым и огонь. «Вода»: пловец и "
                                    "плавсредства."),
        ("Тревога", "Координаты на землю", "Борт не решает — находит, "
                                           "приоритизирует, передаёт."),
    ]
    for i, (k, t, d) in enumerate(boxes):
        x = Inches(0.7) + Inches(4.15) * i
        rect(s, x, Inches(2.15), Inches(3.85), Inches(2.75), fill=WHITE, line=DARK)
        text(s, k.upper(), x + Inches(0.3), Inches(2.35), Inches(3.2), Inches(0.3),
             size=11, bold=True, color=WATER)
        text(s, t, x + Inches(0.3), Inches(2.7), Inches(3.3), Inches(0.6),
             size=21, bold=True, color=DARK)
        text(s, d, x + Inches(0.3), Inches(3.45), Inches(3.3), Inches(1.2),
             size=14, color=GREY, spacing=1.15)
        if i < 2:
            text(s, "→", x + Inches(3.9), Inches(3.2), Inches(0.4), Inches(0.5),
                 size=28, bold=True, color=WATER)
    rect(s, Inches(0.7), Inches(5.25), Inches(11.9), Inches(0.85), fill=LIGHT)
    text(s, "Работает локально, без интернета — в лесу и в зоне ЧС. "
            "Оператор видит очередь объектов, а не поток рамок.",
         Inches(0.95), Inches(5.45), Inches(11.4), Inches(0.5), size=15, color=DARK)
    footer(s, 3)

    # 4 — что уже есть
    s = p(prs)
    header(s, "готовность", "Это не идея — модели уже обучены и проверены")
    big = [("28 из 29", "воздушных кадров с огнём —\nтревога (29 из 29 при "
                        "пониженном пороге)", OK),
           ("97,4%", "кадров с пловцом —\nон обнаружен (1253 из 1286)", WATER),
           ("8,5 мс", "на кадр на видеокарте;\n5–10 кадров/с на борту", DARK)]
    for i, (num, cap, col) in enumerate(big):
        x = Inches(0.7) + Inches(4.15) * i
        text(s, num, x, Inches(2.15), Inches(3.8), Inches(1.0), size=44,
             bold=True, color=col)
        text(s, cap, x, Inches(3.2), Inches(3.7), Inches(1.0), size=14,
             color=GREY, spacing=1.2)
    bullets(s, ["Проверка на данных, которых модели не видели: огонь — "
                "независимый набор, вода — отложенная часть SeaDronesSee.",
                "Код, конфигурации и отчёты открыты — результат можно "
                "воспроизвести и проверить.",
                "Тревога подтверждается 3 кадрами из 5 — а не одним кадром."],
            Inches(0.7), Inches(4.55), Inches(11.9), size=15, gap=0.55)
    footer(s, 4)

    # 5 — что нужно
    s = p(prs)
    header(s, "что нужно от партнёра", "Пять вещей, чтобы проект стал вашим")
    asks = ["Техническое задание на бланке — что искать и что считать успехом",
            "Соглашение о сотрудничестве по форме конкурса ИКаР",
            "Доступ к локации: ландшафт, высоты, сезон, погода, длительность",
            "Одна-две консультации специалистов и фото встречи",
            "Рекомендация о внедрении по итогам испытаний"]
    for i, a in enumerate(asks):
        yy = Inches(2.05) + Inches(0.78) * i
        rect(s, Inches(0.7), yy, Inches(0.48), Inches(0.48), fill=WATER,
             shape=MSO_SHAPE.OVAL)
        text(s, str(i + 1), Inches(0.82), yy + Inches(0.05), Inches(0.3),
             Inches(0.4), size=15, bold=True, color=WHITE)
        text(s, a, Inches(1.4), yy + Inches(0.05), Inches(11.2), Inches(0.5),
             size=17, color=DARK)
    footer(s, 5, "Взамен: рабочий комплекс без лицензионных платежей, "
                 "обучение операторов, дообучение под ваш регион")

    # 6 — контакты
    s = p(prs)
    rect(s, 0, 0, W, H, fill=DARK)
    text(s, "Спасибо за внимание", Inches(0.9), Inches(1.6), Inches(11),
         Inches(1.0), size=40, bold=True, color=WHITE)
    text(s, "Покажем работу моделей на ваших кадрах.", Inches(0.9),
         Inches(2.8), Inches(11), Inches(0.6), size=18,
         color=RGBColor(0xD6, 0xE4, 0xEE))
    rect(s, Inches(0.9), Inches(3.7), Inches(11.5), Inches(0.04), fill=WATER)
    text(s, "Команда «Guslitron774» · ДТ «Кванториум», Челябинск\n"
            "Казаков Кирилл, капитан · [телефон / e-mail]\n"
            "Репозиторий: [ССЫЛКА]",
         Inches(0.9), Inches(4.1), Inches(11.5), Inches(1.6), size=16,
         color=WHITE, spacing=1.4)

    prs.save(OUT)
    print("OK ->", OUT, "| слайдов:", len(prs.slides))


if __name__ == "__main__":
    build()

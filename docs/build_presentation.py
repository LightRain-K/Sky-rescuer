"""Сборка презентации для встречи с предприятием (и защиты проекта).

Запуск:  python docs/build_presentation.py
Выход:   docs/presentation_ikar_bas.pptx  (16:9, ~13 слайдов)
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent / "presentation_ikar_bas.pptx"

DARK = RGBColor(0x1F, 0x3B, 0x57)
GREY = RGBColor(0x44, 0x4A, 0x52)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
FIRE = RGBColor(0xC0, 0x50, 0x2A)
WATER = RGBColor(0x1F, 0x6F, 0x8B)
OK = RGBColor(0x2E, 0x6B, 0x3F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
    s.shadow.inherit = False
    return s


def text(slide, txt, x, y, w, h, size=18, bold=False, color=GREY,
         align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP,
         spacing=1.0, space_after=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    lines = txt.split("\n")
    for i, ln in enumerate(lines):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = spacing
        par.space_after = Pt(space_after)
        r = par.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return tb


def header(slide, kicker, title):
    rect(slide, 0, 0, W, Inches(0.12), fill=DARK)
    text(slide, kicker.upper(), Inches(0.7), Inches(0.42), Inches(11), Inches(0.3),
         size=12, bold=True, color=WATER)
    text(slide, title, Inches(0.7), Inches(0.72), Inches(11.9), Inches(0.9),
         size=30, bold=True, color=DARK)


def footer(slide, n, note=""):
    text(slide, note, Inches(0.7), Inches(6.95), Inches(9), Inches(0.3),
         size=10, color=RGBColor(0x8A, 0x8A, 0x8A))
    text(slide, str(n), Inches(12.3), Inches(6.95), Inches(0.5), Inches(0.3),
         size=10, color=RGBColor(0x8A, 0x8A, 0x8A), align=PP_ALIGN.RIGHT)


def bullets(slide, items, x, y, w, size=18, gap=0.55, color=GREY):
    for i, it in enumerate(items):
        yy = y + Inches(gap * i)
        rect(slide, x, yy + Inches(0.1), Inches(0.09), Inches(0.09), fill=WATER,
             shape=MSO_SHAPE.OVAL)
        text(slide, it, x + Inches(0.28), yy, w, Inches(0.5), size=size, color=color)


def table(slide, rows, x, y, w, col_w, size=13, header_fill=DARK, row_h=0.34):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w,
                                   Inches(row_h * len(rows)))
    tbl = shape.table
    for ci, cw in enumerate(col_w):
        tbl.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08)
            par = cell.text_frame.paragraphs[0]
            r = par.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = "Calibri"
            if ri == 0:
                r.font.bold = True
                r.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            else:
                r.font.color.rgb = GREY
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
    return tbl


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1 — титул
    s = blank(prs); n += 1
    rect(s, 0, 0, W, H, fill=DARK)
    rect(s, 0, Inches(4.55), W, Inches(0.06), fill=WATER)
    text(s, "Инженерные кадры России · ИКаР-БАС · сезон 2026/27",
         Inches(0.9), Inches(0.8), Inches(11), Inches(0.4), size=14,
         color=RGBColor(0x9F, 0xC5, 0xD8))
    text(s, "Самолётный беспилотный комплекс\nс бортовым искусственным интеллектом",
         Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.8), size=40, bold=True,
         color=WHITE, spacing=1.05)
    text(s, "Поиск лесных пожаров и людей, которым нужна помощь на воде",
         Inches(0.9), Inches(3.4), Inches(11), Inches(0.6), size=22,
         color=RGBColor(0xD6, 0xE4, 0xEE))
    text(s, "Команда [НАЗВАНИЕ]  ·  ДТ «Кванториум», Челябинск\n"
            "Предприятие-партнёр: [НАЗВАНИЕ]",
         Inches(0.9), Inches(4.9), Inches(11), Inches(1.0), size=16, color=WHITE)

    # 2 — проблема
    s = blank(prs); n += 1
    header(s, "проблема", "Патрулирование не успевает за пожаром")
    col_w, gap = Inches(3.8), Inches(0.45)
    cards = [
        ("Часы на маршрут", "Выход на маршрут и осмотр десятков тысяч гектаров "
                            "занимают часы. Дым замечают, когда очаг уже набрал силу."),
        ("Коптер — 20–30 минут", "Малый дрон не покрывает большой район: "
                                 "время в воздухе — главный ограничитель."),
        ("Человек в воде — 20 пикселей", "С берега его видно плохо, а счёт "
                                         "идёт на минуты. Люди ищут глазами."),
    ]
    for i, (t, d) in enumerate(cards):
        x = Inches(0.7) + (col_w + gap) * i
        rect(s, x, Inches(2.05), col_w, Inches(3.6), fill=LIGHT)
        rect(s, x, Inches(2.05), Inches(0.08), Inches(3.6), fill=FIRE)
        text(s, t, x + Inches(0.32), Inches(2.35), col_w - Inches(0.55), Inches(0.7),
             size=20, bold=True, color=DARK)
        text(s, d, x + Inches(0.32), Inches(3.15), col_w - Inches(0.55), Inches(2.2),
             size=15, color=GREY, spacing=1.15)
    text(s, "Спутник даёт редкие снимки с крупным пикселем. Наблюдатель зависит "
            "от погоды и внимания. Нужен непрерывный обзор с воздуха — и время в полёте.",
         Inches(0.7), Inches(5.95), Inches(12), Inches(0.7), size=16, color=DARK)
    footer(s, n)

    # 3 — решение
    s = blank(prs); n += 1
    header(s, "решение", "Самолёт даёт время, ИИ — непрерывное внимание")
    boxes = [
        ("Носитель", "Самолётный БПЛА", "Часы в воздухе вместо десятков минут. "
                                        "Тот же автопилот, та же камера."),
        ("Интеллект", "Две модели по профилю", "«Лес»: дым и огонь. «Вода»: "
                                               "пловец и плавсредства. Не мешают друг другу."),
        ("Тревога", "Координаты на землю", "Борт не решает — он находит, "
                                           "приоритизирует и передаёт оператору."),
    ]
    for i, (k, t, d) in enumerate(boxes):
        x = Inches(0.7) + Inches(4.15) * i
        rect(s, x, Inches(2.1), Inches(3.85), Inches(3.3), fill=WHITE, line=DARK)
        text(s, k.upper(), x + Inches(0.3), Inches(2.3), Inches(3.2), Inches(0.3),
             size=11, bold=True, color=WATER)
        text(s, t, x + Inches(0.3), Inches(2.65), Inches(3.25), Inches(0.8),
             size=21, bold=True, color=DARK)
        text(s, d, x + Inches(0.3), Inches(3.5), Inches(3.25), Inches(1.7),
             size=14, color=GREY, spacing=1.15)
        if i < 2:
            text(s, "→", x + Inches(3.9), Inches(3.4), Inches(0.4), Inches(0.5),
                 size=28, bold=True, color=WATER)
    text(s, "Локальный инференс: работает без интернета — в лесу и в зоне ЧС.",
         Inches(0.7), Inches(5.8), Inches(12), Inches(0.5), size=16, bold=True, color=DARK)
    footer(s, n)

    # 4 — пожарная модель
    s = blank(prs); n += 1
    header(s, "что уже сделано · лес", "Модель находит огонь и дым на чужих данных")
    table(s, [["Класс", "Precision", "Recall", "mAP50"],
              ["Огонь", "0,715", "0,655", "0,670"],
              ["Дым", "0,831", "0,772", "0,797"]],
          Inches(0.7), Inches(2.0), Inches(6.1), [2.8, 1.1, 1.1, 1.1], size=15)
    text(s, "Проверка: D-Fire test, 4306 кадров.\nМодель их не видела при обучении.",
         Inches(0.7), Inches(3.5), Inches(6.2), Inches(0.9), size=14, color=GREY)
    rect(s, Inches(7.2), Inches(1.95), Inches(5.4), Inches(2.65), fill=LIGHT)
    text(s, "28 из 29", Inches(7.5), Inches(2.25), Inches(4.8), Inches(0.9),
         size=44, bold=True, color=OK)
    text(s, "воздушных кадров с огнём — тревога\nпри рабочем пороге (29 из 29 при пониженном)",
         Inches(7.5), Inches(3.25), Inches(4.8), Inches(1.1), size=15, color=GREY, spacing=1.15)
    bullets(s, ["На кадрах без огня — ложных тревог нет.",
                "Порог подтверждения: 3 из 5 кадров, а не один кадр.",
                "Обучение: открытые наборы + собственные кадры с дрона."],
            Inches(0.7), Inches(4.55), Inches(11.8), size=16, gap=0.5)
    footer(s, n)

    # 5 — водная модель
    s = blank(prs); n += 1
    header(s, "что уже сделано · вода", "Модель находит человека в воде")
    table(s, [["Класс", "Precision", "Recall", "mAP50"],
              ["Пловец", "0,840", "0,803", "0,759"],
              ["Лодка", "0,933", "0,939", "0,939"],
              ["Гидроцикл", "0,870", "0,884", "0,858"],
              ["Буй", "0,863", "0,777", "0,814"]],
          Inches(0.7), Inches(1.95), Inches(6.1), [2.8, 1.1, 1.1, 1.1], size=14)
    text(s, "Проверка: отложенная часть SeaDronesSee\n(съёмка с дронов над водой).",
         Inches(0.7), Inches(4.0), Inches(6.2), Inches(0.9), size=14, color=GREY)
    rect(s, Inches(7.2), Inches(1.95), Inches(5.4), Inches(2.65), fill=LIGHT)
    text(s, "97,4%", Inches(7.5), Inches(2.25), Inches(4.8), Inches(0.9),
         size=44, bold=True, color=WATER)
    text(s, "кадров, где пловец присутствует, —\nон обнаружен (1253 из 1286)",
         Inches(7.5), Inches(3.25), Inches(4.8), Inches(1.1), size=15, color=GREY, spacing=1.15)
    bullets(s, ["Пловца не путает с плавсредством на отложенных данных.",
                "Отдельно проверено: на суше модель молчит — профили раздельные.",
                "Класс определяется по истории трека, а не по одному кадру."],
            Inches(0.7), Inches(4.85), Inches(11.8), size=15, gap=0.45)
    footer(s, n)

    # 6 — триаж
    s = blank(prs); n += 1
    header(s, "триаж", "Оператор получает очередь, а не сто точек")
    left = ["Дрейфует — не гребёт к безопасности", "Далеко от лодок — рядом никого",
            "Долго в воде — время работает против"]
    for i, t in enumerate(left):
        yy = Inches(2.05) + Inches(0.85) * i
        rect(s, Inches(0.7), yy, Inches(5.3), Inches(0.7), fill=LIGHT)
        text(s, f"+1   {t}", Inches(0.95), yy + Inches(0.15), Inches(4.9), Inches(0.45),
             size=15, color=DARK)
    text(s, "Приоритет = сумма признаков\n(0…3)", Inches(0.7), Inches(4.7),
         Inches(5.3), Inches(0.9), size=15, bold=True, color=DARK, spacing=1.15)
    rect(s, Inches(6.5), Inches(2.05), Inches(6.1), Inches(3.55), fill=WHITE, line=WATER)
    text(s, "Очередь у оператора", Inches(6.8), Inches(2.25), Inches(5.5), Inches(0.4),
         size=13, bold=True, color=WATER)
    rows = [("1", "пловец · дрейф, до лодки 460 px, 61 кадр", "3"),
            ("2", "пловец · в воде 633 кадра", "2"),
            ("3", "лодка · стоит на месте — проверить", "—")]
    for i, (rank, desc, score) in enumerate(rows):
        yy = Inches(2.75) + Inches(0.85) * i
        text(s, rank, Inches(6.8), yy, Inches(0.4), Inches(0.4), size=18, bold=True, color=DARK)
        text(s, desc, Inches(7.3), yy, Inches(4.3), Inches(0.5), size=13, color=GREY)
        text(s, score, Inches(11.7), yy, Inches(0.6), Inches(0.4), size=18,
             bold=True, color=FIRE, align=PP_ALIGN.RIGHT)
    text(s, "Проверено на видео из открытых источников: одиночный пловец "
            "удерживается одним треком 433 кадра и выходит в верх очереди.",
         Inches(0.7), Inches(5.85), Inches(12), Inches(0.6), size=14, color=GREY)
    footer(s, n)

    # 7 — архитектура
    s = blank(prs); n += 1
    header(s, "архитектура", "Как это работает на борту")
    y = Inches(2.5)
    steps = [("Камера", "кадр 960×960", 0.7), ("Профиль миссии", "«лес» / «вода»", 3.35),
             ("Нейросеть", "детекция", 6.0), ("3 из 5 кадров", "подтверждение", 8.65)]
    for t, d, x in steps:
        rect(s, Inches(x), y, Inches(2.25), Inches(1.35), fill=LIGHT)
        text(s, t, Inches(x) + Inches(0.15), y + Inches(0.18), Inches(1.95), Inches(0.4),
             size=16, bold=True, color=DARK, align=PP_ALIGN.CENTER)
        text(s, d, Inches(x) + Inches(0.15), y + Inches(0.65), Inches(1.95), Inches(0.4),
             size=12, color=GREY, align=PP_ALIGN.CENTER)
    for x in (2.95, 5.6, 8.25):
        text(s, "→", Inches(x), y + Inches(0.45), Inches(0.4), Inches(0.5),
             size=24, bold=True, color=WATER)
    rect(s, Inches(0.7), Inches(4.35), Inches(11.9), Inches(1.25), fill=WHITE, line=WATER)
    text(s, "Трекер → приоритет → координаты → тревога на наземный пункт",
         Inches(0.95), Inches(4.55), Inches(11.4), Inches(0.5), size=18, bold=True, color=DARK)
    text(s, "Плюс: запись трека и обоснование приоритета — оператор видит, "
            "почему объект в топе", Inches(0.95), Inches(5.0), Inches(11.4), Inches(0.4),
         size=13, color=GREY)
    text(s, "Raspberry Pi 5 · без интернета · 5–10 кадров/с",
         Inches(0.7), Inches(5.95), Inches(12), Inches(0.5), size=16, bold=True, color=DARK)
    footer(s, n)

    # 8 — честные границы
    s = blank(prs); n += 1
    header(s, "важно для службы", "Что система НЕ делает")
    bullets(s, ["Не ставит диагноз «тонет». По одному кадру с высоты это невозможно — "
                "и мы этого не обещаем.",
                "Не принимает решений и не действует автономно: передаёт координаты "
                "и приоритет.",
                "Не заменяет оператора. Сокращает время поиска и объём работы."],
            Inches(0.7), Inches(2.0), Inches(11.9), size=18, gap=0.75)
    rect(s, Inches(0.7), Inches(4.35), Inches(11.9), Inches(1.5), fill=LIGHT)
    text(s, "Почему так честнее: ложное обещание «ИИ найдёт утопающего» дорого "
            "стоит на реальном выезде.\nНаша роль — быстрее найти, показать и "
            "подсказать, кому помочь первым.",
         Inches(0.95), Inches(4.6), Inches(11.4), Inches(1.1), size=15, color=DARK, spacing=1.2)
    footer(s, n)

    # 9 — что уже проверено
    s = blank(prs); n += 1
    header(s, "готовность", "Что уже есть, кроме идеи")
    items = [
        ("Две обученные модели", "с метриками на данных, которых они не видели"),
        ("Воспроизводимый тест", "одна команда — отчёт с метриками сохраняется"),
        ("Модуль триажа", "проверен на видео, класс не «мерцает»"),
        ("Открытый репозиторий", "код, конфигурации, отчёты, история работы"),
        ("Оценка скорости", "8,5 мс/кадр на видеокарте; 5–10 кадров/с на борту"),
    ]
    for i, (t, d) in enumerate(items):
        yy = Inches(1.95) + Inches(0.85) * i
        rect(s, Inches(0.7), yy, Inches(0.55), Inches(0.55), fill=OK,
             shape=MSO_SHAPE.OVAL)
        text(s, "✓", Inches(0.83), yy + Inches(0.07), Inches(0.35), Inches(0.4),
             size=18, bold=True, color=WHITE)
        text(s, t, Inches(1.45), yy + Inches(0.03), Inches(4.6), Inches(0.4),
             size=17, bold=True, color=DARK)
        text(s, d, Inches(6.2), yy + Inches(0.05), Inches(6.3), Inches(0.4),
             size=14, color=GREY)
    footer(s, n, "Ссылка на репозиторий и QR-код — на последнем слайде")

    # 10 — план
    s = blank(prs); n += 1
    header(s, "план", "Что делаем дальше — вместе с предприятием")
    table(s, [["Этап", "Результат", "Срок"],
              ["ТЗ и условия работы", "согласованные требования", "[дата]"],
              ["Съёмка на локации", "кадры региона, разметка", "[дата]"],
              ["Дообучение моделей", "устойчивость к местным условиям", "[дата]"],
              ["Сборка и настройка борта", "летающий комплекс", "[дата]"],
              ["Испытания", "протокол, замер дальности и качества", "[дата]"],
              ["Предзащита перед предприятием", "рекомендация о внедрении", "[дата]"]],
          Inches(0.7), Inches(2.0), Inches(11.9), [4.6, 5.4, 1.9], size=14, row_h=0.42)
    text(s, "Сроки заочного этапа: материалы — за 30 дней до очного; "
            "реестр команд — до 31.12.2026.",
         Inches(0.7), Inches(5.6), Inches(12), Inches(0.5), size=14, color=GREY)
    footer(s, n)

    # 11 — что просим
    s = blank(prs); n += 1
    header(s, "что нужно от предприятия", "Четыре документа и одна локация")
    asks = [
        ("Техническое задание", "на бланке, с подписью — какие объекты искать, "
                                "в каких условиях, что считать успехом"),
        ("Соглашение о сотрудничестве", "по форме конкурса ИКаР, сроки — на сезон"),
        ("Рекомендация о внедрении", "по итогам испытаний на вашей локации"),
        ("Консультации специалистов", "1–2 встречи и фото — это отдельная номинация"),
    ]
    for i, (t, d) in enumerate(asks):
        yy = Inches(1.95) + Inches(0.95) * i
        rect(s, Inches(0.7), yy, Inches(0.5), Inches(0.5), fill=WATER, shape=MSO_SHAPE.OVAL)
        text(s, str(i + 1), Inches(0.83), yy + Inches(0.05), Inches(0.3), Inches(0.4),
             size=16, bold=True, color=WHITE)
        text(s, t, Inches(1.4), yy, Inches(4.5), Inches(0.45), size=17, bold=True, color=DARK)
        text(s, d, Inches(6.1), yy + Inches(0.02), Inches(6.5), Inches(0.6),
             size=14, color=GREY)
    rect(s, Inches(0.7), Inches(5.85), Inches(11.9), Inches(0.75), fill=LIGHT)
    text(s, "Плюс доступ к локации: ландшафт, высоты, сезон, погода, "
            "требуемая длительность — это обязательный раздел паспорта.",
         Inches(0.95), Inches(6.05), Inches(11.4), Inches(0.4), size=14, color=DARK)
    footer(s, n)

    # 12 — что даём
    s = blank(prs); n += 1
    header(s, "что получает предприятие", "Инструмент, а не отчёт")
    bullets(s, ["Рабочий комплекс без лицензионных платежей — код открыт.",
                "Обучение операторов: тревога = точка на карте.",
                "Дообучение под ваш регион на ваших кадрах — без нас.",
                "Честные границы и протоколы испытаний вместо обещаний.",
                "Возможность масштабирования: второй борт, тепловизор для ночи, "
                "карта очагов."],
            Inches(0.7), Inches(2.0), Inches(11.9), size=18, gap=0.72)
    footer(s, n)

    # 13 — контакты
    s = blank(prs); n += 1
    rect(s, 0, 0, W, H, fill=DARK)
    text(s, "Спасибо за внимание", Inches(0.9), Inches(1.5), Inches(11), Inches(1.0),
         size=40, bold=True, color=WHITE)
    text(s, "Готовы ответить на вопросы и показать работу моделей на ваших кадрах.",
         Inches(0.9), Inches(2.7), Inches(11), Inches(0.6), size=18,
         color=RGBColor(0xD6, 0xE4, 0xEE))
    rect(s, Inches(0.9), Inches(3.6), Inches(11.5), Inches(0.04), fill=WATER)
    text(s, "Команда [НАЗВАНИЕ] · ДТ «Кванториум», Челябинск\n"
            "Капитан: Казаков Кирилл · [телефон / e-mail]\n"
            "Репозиторий: [ССЫЛКА]",
         Inches(0.9), Inches(4.0), Inches(11.5), Inches(1.6), size=16, color=WHITE,
         spacing=1.4)
    text(s, "[QR-КОД НА РЕПОЗИТОРИЙ]", Inches(10.4), Inches(4.1), Inches(2.2),
         Inches(1.2), size=12, color=RGBColor(0x9F, 0xC5, 0xD8))

    prs.save(OUT)
    print("OK ->", OUT, "| слайдов:", len(prs.slides))


if __name__ == "__main__":
    build()

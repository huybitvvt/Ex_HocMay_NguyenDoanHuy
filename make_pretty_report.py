import html
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


GITHUB_LINK = "https://github.com/huybitvvt/Ex_HocMay_NguyenDoanHuy.git"
RESULTS_CSV = "optimization_results.csv"
CURVES_IMG = "training_curves.png"
PDF_PATH = "optimization_cifar10_NguyenDoanHuy.pdf"


def register_fonts():
    candidates = [
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if os.path.exists(regular) and os.path.exists(bold):
            pdfmetrics.registerFont(TTFont("VN-Regular", regular))
            pdfmetrics.registerFont(TTFont("VN-Bold", bold))
            return "VN-Regular", "VN-Bold"
    return "Helvetica", "Helvetica-Bold"


def p(text, style):
    return Paragraph(html.escape(str(text)), style)


def build_report():
    regular_font, bold_font = register_fonts()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleVN",
        fontName=bold_font,
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#1F2937"),
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SubVN",
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#374151"),
        alignment=TA_CENTER,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="HeadingVN",
        fontName=bold_font,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#111827"),
        spaceBefore=8,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyVN",
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="LinkVN",
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#111827"),
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CellVN",
        fontName=regular_font,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#111827"),
    ))
    styles.add(ParagraphStyle(
        name="CellBoldVN",
        fontName=bold_font,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#111827"),
        alignment=TA_CENTER,
    ))

    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)
    best = df.iloc[0]

    table_df = pd.DataFrame({
        "Mô hình": df["experiment"],
        "Optimizer": df["optimizer"],
        "BN": df["batch_norm"],
        "Dropout": df["dropout"].map(lambda x: f"{x:.1f}"),
        "Train loss": df["final_train_loss"].map(lambda x: f"{x:.4f}"),
        "Best val acc": df["best_val_accuracy"].map(lambda x: f"{x * 100:.2f}%"),
        "Final val acc": df["final_val_accuracy"].map(lambda x: f"{x * 100:.2f}%"),
        "Time": df["train_time_sec"].map(lambda x: f"{x:.1f}s"),
        "Conv epoch": df["convergence_epoch"].astype(int),
    })

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(A4),
        rightMargin=1.1 * cm,
        leftMargin=1.1 * cm,
        topMargin=0.9 * cm,
        bottomMargin=0.9 * cm,
        title="Báo cáo Optimization CIFAR-10",
        author="Nguyễn Đoàn Huy",
    )

    story = []
    story.append(Paragraph("Báo cáo Optimization với CNN trên CIFAR-10", styles["TitleVN"]))
    story.append(Paragraph(
        "Bài thực hành so sánh ảnh hưởng của Batch Normalization, Dropout và các optimizer đối với bài toán phân loại ảnh CIFAR-10.",
        styles["SubVN"],
    ))

    safe_link = html.escape(GITHUB_LINK)
    story.append(Paragraph(
        f'Link code GitHub: <link href="{safe_link}"><font color="#0563C1"><u>{safe_link}</u></font></link>',
        styles["LinkVN"],
    ))
    story.append(Paragraph(
        "Chế độ chạy: FULL RUN; số epoch: 20; batch size: 128; device: cuda",
        styles["SubVN"],
    ))

    story.append(Paragraph("B3. Bảng so sánh", styles["HeadingVN"]))
    data = [[p(col, styles["CellBoldVN"]) for col in table_df.columns]]
    for _, row in table_df.iterrows():
        data.append([p(value, styles["CellVN"]) for value in row])

    table = Table(
        data,
        repeatRows=1,
        colWidths=[4.0 * cm, 3.1 * cm, 1.2 * cm, 1.6 * cm, 2.0 * cm, 2.3 * cm, 2.3 * cm, 1.8 * cm, 2.0 * cm],
        hAlign="CENTER",
    )
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBFF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for row_idx in range(1, len(data)):
        if row_idx == 1:
            table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#EAF7EA")))
        elif row_idx % 2 == 0:
            table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#F9FAFB")))
    table.setStyle(TableStyle(table_style))
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))

    if os.path.exists(CURVES_IMG):
        story.append(Image(CURVES_IMG, width=23.8 * cm, height=8.1 * cm))

    story.append(Paragraph("B4. Kết luận", styles["HeadingVN"]))
    conclusion = (
        f"Sau khi chạy các thí nghiệm, em thấy cấu hình tốt nhất là {best['experiment']}, sử dụng optimizer {best['optimizer']}, "
        f"Batch Normalization: {best['batch_norm']} và Dropout = {best['dropout']:.1f}. "
        f"Cấu hình này đạt validation accuracy cao nhất là {best['best_val_accuracy'] * 100:.2f}%, "
        f"training loss cuối là {best['final_train_loss']:.4f}, hội tụ ở epoch {int(best['convergence_epoch'])} "
        f"và thời gian train khoảng {best['train_time_sec']:.1f} giây. "
        "Kết quả cho thấy Batch Normalization giúp quá trình học ổn định hơn so với CNN cơ bản, "
        "Dropout giúp hạn chế overfitting, còn Adam cho khả năng cải thiện nhanh và đạt accuracy cao nhất trong các cấu hình đã thử."
    )
    story.append(Paragraph(conclusion, styles["BodyVN"]))

    doc.build(story)


if __name__ == "__main__":
    build_report()
    print(f"Đã tạo lại file: {PDF_PATH}")

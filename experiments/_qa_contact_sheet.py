from pathlib import Path

from PIL import Image, ImageDraw


folder = Path("outputs/direct_rx_rnj_analysis")
names = [
    "true_R_X_heatmaps.png",
    "fitted_R_X_RX75_heatmaps.png",
    "true_R_X_sorted_pair_bars.png",
    "fitted_R_X_RX75_sorted_pair_bars.png",
]
panels = []
for name in names:
    image = Image.open(folder / name).convert("RGB")
    image.thumbnail((700, 700))
    panel = Image.new("RGB", (720, image.height + 40), "white")
    panel.paste(image, ((720 - image.width) // 2, 35))
    ImageDraw.Draw(panel).text((10, 10), name, fill="black")
    panels.append(panel)
sheet = Image.new("RGB", (720, sum(panel.height for panel in panels)), "white")
y = 0
for panel in panels:
    sheet.paste(panel, (0, y))
    y += panel.height
sheet.save(folder / "qa_contact_sheet.jpg", quality=72, optimize=True)

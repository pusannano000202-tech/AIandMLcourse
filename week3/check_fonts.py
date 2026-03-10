import matplotlib.font_manager as fm

print("=== Available Fonts Check ===")
font_list = [f.name for f in fm.fontManager.ttflist]
korean_fonts = ['Malgun Gothic', 'Gulim', 'Batang', 'Dotum', 'NanumGothic', 'AppleGothic']

found = False
for font in korean_fonts:
    if font in font_list:
        print(f"Found Korean font: {font}")
        found = True

if not found:
    print("No standard Korean fonts found in matplotlib font manager.")
    print("Top 10 available fonts:")
    for f in font_list[:10]:
        print(f"- {f}")
else:
    print("You can use one of the found fonts.")

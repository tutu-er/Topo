"""Render every page and create labeled six-page contact sheets for visual QA."""
from pathlib import Path
import argparse, subprocess
from PIL import Image, ImageDraw
from pypdf import PdfReader
ap=argparse.ArgumentParser()
ap.add_argument('pdf'); ap.add_argument('out'); ap.add_argument('--dpi',type=int,default=90)
a=ap.parse_args(); pdf=Path(a.pdf).resolve(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
subprocess.run(['pdftoppm','-r',str(a.dpi),'-png',str(pdf),str(out/'page')],check=True,stdout=subprocess.DEVNULL)
files=sorted(out.glob('page-*.png'))
assert len(files)==len(PdfReader(str(pdf)).pages)
for start in range(0,len(files),6):
    canvas=Image.new('RGB',(1000,2184),'#d8dfe8'); d=ImageDraw.Draw(canvas)
    for pos,f in enumerate(files[start:start+6]):
        im=Image.open(f).convert('RGB'); im.thumbnail((478,696))
        x=(pos%2)*500+(500-im.width)//2; y=(pos//2)*728+25
        canvas.paste(im,(x,y)); d.text(((pos%2)*500+12,(pos//2)*728+7),f'PDF page {start+pos+1}',fill='black')
    canvas.save(out/f'contact-{start//6+1:02d}.jpg',quality=88)
print(f'{len(files)} pages rendered to {out}')

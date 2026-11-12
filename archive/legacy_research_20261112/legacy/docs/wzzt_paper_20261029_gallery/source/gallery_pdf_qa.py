"""PDF count/text/render checks and optional figure-book assembly."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime,timezone
from pypdf import PdfReader,PdfWriter
from PIL import Image,ImageOps,ImageDraw


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(directory,rebuild_book=False):
    p=Path(directory).resolve();review=p/'review';review.mkdir(exist_ok=True)
    with (p/'figure_index.csv').open(encoding='utf-8') as stream:figures=list(csv.DictReader(stream))
    if rebuild_book:
        writer=PdfWriter()
        for row in figures:writer.append(str(p/row['pdf']))
        writer.add_metadata({'/Title':'WZZT recorded-data figure collection'})
        with (p/'all_figures.pdf').open('wb') as stream:writer.write(stream)
    pages=PdfReader(p/'all_figures.pdf').pages;records=[]
    assert len(pages)==len(figures)==36
    for i,row in enumerate(figures):
        reader=PdfReader(p/row['pdf']);text=reader.pages[0].extract_text() or ''
        assert len(reader.pages)==1 and len(text)>20
        assert len(pages[i].extract_text() or '')>20
        records.append(dict(figure_id=row['figure_id'],page=i+1,pdf_sha256=digest(p/row['pdf']),extracted_characters=len(text)))
    executable=shutil.which('pdftoppm')
    if executable is None:raise RuntimeError('Poppler pdftoppm unavailable')
    version=subprocess.run([executable,'-v'],capture_output=True,text=True).stderr.splitlines()[0]
    subprocess.run([executable,'-scale-to-x','1400','-scale-to-y','-1','-png',str(p/'all_figures.pdf'),str(review/'pdf_page')],check=True)
    rendered=sorted(review.glob('pdf_page-*.png'));assert len(rendered)==36
    for i,file in enumerate(rendered):
        with Image.open(file) as im:im.verify()
        records[i]['rendered_sha256']=digest(file)
    for offset in range(0,36,6):
        canvas=Image.new('RGB',(2400,2400),'#eef1f5');draw=ImageDraw.Draw(canvas)
        for j,file in enumerate(rendered[offset:offset+6]):
            with Image.open(file) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(1180,735));x=(j%2)*1200+10+(1180-thumb.width)//2;y=(j//2)*800+40+(735-thumb.height)//2;canvas.paste(thumb,(x,y))
            draw.text(((j%2)*1200+15,(j//2)*800+10),records[offset+j]['figure_id'],fill='black')
        canvas.save(review/f'pdf_sheet_{offset//6+1:02d}.png')
    receipt=dict(all_passed=True,pages=36,individual_PDFs=36,renderer=version,utc=datetime.now(timezone.utc).isoformat(),
        all_figures_sha256=digest(p/'all_figures.pdf'),checks=records,visual_inspection='pending; this receipt verifies extraction and rendering only')
    (p/'pdf_render_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PDF page counts, text and rendering passed: 36 pages / 36 individual PDFs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--rebuild-book',action='store_true');args=parser.parse_args();run(args.directory,args.rebuild_book)

"""Compile the existing report and prepare v9 render/content verification."""
import json
from pathlib import Path
import re

import verify_report_pdf_v7 as verifier


if __name__ == '__main__':
    verifier.RESULTS = Path(__file__).resolve().parent / 'results_v9'
    verifier.main()
    target = verifier.RESULTS / 'report_pdf_verification.json'
    result = json.loads(target.read_text(encoding='utf-8'))
    text = (Path(result['review_directory']) / 'extracted_text.txt').read_text(encoding='utf-8')
    flat = re.sub(r'\s+', '', text)
    result['v9_content_checks'] = {phrase: phrase in flat for phrase in (
        '当前算法总览', '固定增益的线性外表示', '同一个初始候选库',
        '23个节点', '27kWh', '原AC竞争见证', '两个最终外LP不保证集合嵌套')}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if not all(result['v9_content_checks'].values()):
        raise RuntimeError('v9 PDF content checks require attention')

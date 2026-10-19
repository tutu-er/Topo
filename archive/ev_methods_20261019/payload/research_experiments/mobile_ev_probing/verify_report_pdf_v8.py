"""Reuse the established PDF verifier for the existing report, store v8 evidence."""
import json
from pathlib import Path
import re

import verify_report_pdf_v7 as verifier


if __name__ == '__main__':
    verifier.RESULTS = Path(__file__).resolve().parent / 'results_v8'
    verifier.main()
    target = verifier.RESULTS / 'report_pdf_verification.json'
    result = json.loads(target.read_text(encoding='utf-8'))
    text = (Path(result['review_directory'])/'extracted_text.txt').read_text(encoding='utf-8')
    flat = re.sub(r'\s+', '', text)
    checks = {phrase:phrase in flat for phrase in ('被动优先', 'AC补证', '162kWh', '2598', '共同源')}
    result['v8_content_checks'] = checks
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if not all(checks.values()):
        raise RuntimeError('v8 PDF content checks require attention')

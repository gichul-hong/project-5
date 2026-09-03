import json
from pathlib import Path

for p in [Path('claude-final/iter7.ipynb'), Path('claude/iter7.ipynb')]:
    nb = json.load(p.open(encoding='utf-8'))
    
    # Update config.epochs to 3
    for i, line in enumerate(nb['cells'][4]['source']):
        if 'epochs: int =' in line:
            nb['cells'][4]['source'][i] = '    epochs: int = 3            # 전체 데이터(3167장) 사용 시 1에폭당 16분 소요되므로 3에폭(약 48분) 설정\n'
    
    # Update SPECS epochs comment
    for c in nb['cells']:
        if 'SPECS: dict' in ''.join(c.get('source', [])):
            for i, line in enumerate(c['source']):
                if 'epochs": config.epochs' in line:
                    c['source'][i] = '        "epochs": config.epochs,      # 3 Epochs (전체 데이터 기준 약 48분 소요)\n'

    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Updated {p} to 3 epochs for full dataset!')

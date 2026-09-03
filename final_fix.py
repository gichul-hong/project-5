import json
from pathlib import Path

for p in [Path('claude-final/iter7.ipynb'), Path('claude/iter7.ipynb')]:
    nb = json.load(p.open(encoding='utf-8'))
    
    # 1. Update config
    for i, line in enumerate(nb['cells'][4]['source']):
        if 'batch_size: int =' in line:
            nb['cells'][4]['source'][i] = '    batch_size: int = 16       # 5-Stage U-Net OOM 방지를 위한 최적값\n'
        if 'epochs: int =' in line:
            nb['cells'][4]['source'][i] = '    epochs: int = 3            # 전체 데이터(3167장) 사용 시 1에폭당 10분 이상 소요되므로 3에폭(약 30~40분) 설정\n'
    
    # 2. Update SPECS epochs comment
    for c in nb['cells']:
        if 'SPECS: dict' in ''.join(c.get('source', [])):
            for i, line in enumerate(c['source']):
                if 'epochs": config.epochs' in line:
                    c['source'][i] = '        "epochs": config.epochs,      # 3 Epochs (전체 데이터 기준 1시간 이내)\n'

    # Check if FP16 NaN patch is present. (It should be from my previous python one-liner, but if they reloaded, I need to make sure)
    patch_sg = '''class SimpleGate(nn.Module):
    # NAFNet: 비선형 활성화 함수(GELU 등)를 대체하는 단순 채널 분할 곱셈 (FP16 Overflow 방지 패치)
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        if x.dtype == torch.float16:
            return (x1.float() * x2.float()).half()
        return x1 * x2'''
        
    patch_sca = '''class SimplifiedChannelAttention(nn.Module):
    # NAFNet: 극경량 채널 어텐션 (O(1) complexity per channel) (FP16 Overflow 방지 패치)
    def __init__(self, c: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1, 1, 0)
    def forward(self, x: Tensor) -> Tensor:
        att = self.conv(self.pool(x))
        if x.dtype == torch.float16:
            return (x.float() * att.float()).half()
        return x * att'''

    for c in nb['cells']:
        if 'class SimpleGate' in ''.join(c.get('source', [])):
            src = ''.join(c['source'])
            if 'x1.float() * x2.float()' not in src:
                # 1. Patch SimpleGate
                sg_start = src.find('class SimpleGate')
                sg_end = src.find('class SimplifiedChannelAttention')
                if sg_start != -1 and sg_end != -1:
                    src = src[:sg_start] + patch_sg + '\n\n\n' + src[sg_end:]
                    
                # 2. Patch SCA
                sca_start = src.find('class SimplifiedChannelAttention')
                sca_end = src.find('class NAFBlock')
                if sca_start != -1 and sca_end != -1:
                    src = src[:sca_start] + patch_sca + '\n\n\n' + src[sca_end:]
                    
                # 3. Patch DC rho
                src = src.replace('rho = self.log_rho.exp().clamp(1e-4, 10.0)', 'rho = self.log_rho.float().clamp(-9.0, 2.5).exp()')
                
                # 4. Patch eps in sqrt
                src = src.replace('+ 1e-12)', '+ 1e-6)')
                
                c['source'] = [line + '\n' for line in src.split('\n')]
                c['source'][-1] = c['source'][-1].rstrip('\n')

    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Updated {p} to Batch 16, 3 epochs, and FP16 NaN safety!')

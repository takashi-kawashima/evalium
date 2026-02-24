import glob
import pandas as pd

files = glob.glob('data/**/**/*.xlsx', recursive=True) + glob.glob('data/**/*.xlsx')
files = list(dict.fromkeys(files))
if not files:
    print('no excel files found')
    raise SystemExit(1)

for f in files:
    print('---', f)
    try:
        df = pd.read_excel(f)
        print('columns:', list(df.columns))
        print(df.head().to_markdown(index=False))
    except Exception as e:
        print(' read error', e)

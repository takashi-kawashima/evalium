import sys
def check(name):
    try:
        __import__(name)
        print(name, 'OK')
    except Exception as e:
        print(name, 'MISSING or import error:', e)

for pkg in ['langchain_openai', 'langchain', 'openai', 'tabulate', 'pandas']:
    check(pkg)

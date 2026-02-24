import traceback

def try_import(name, attr=None):
    try:
        if attr:
            mod = __import__(name, fromlist=[attr])
            print(f"import {name}.{attr} OK")
        else:
            __import__(name)
            print(f"import {name} OK")
    except Exception as e:
        print(f"import {name}{'.'+attr if attr else ''} FAILED")
        traceback.print_exc()

try_import('langchain_openai', 'OpenAIEmbeddings')
try_import('langchain_openai', 'AzureOpenAIEmbeddings')
try_import('langchain.embeddings', 'OpenAIEmbeddings')

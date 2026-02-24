from dotenv import load_dotenv
load_dotenv('local.env')
import os, traceback
texts = ['テスト']
key = os.getenv('RAKUTEN_AI_GATEWAY_KEY')
base_openai = os.getenv('RAKUTEN_AI_GATEWAY_BASE','https://api.ai.public.rakuten-it.com/rakutenllms/v1/')
base_azure = os.getenv('RAKUTEN_AI_GATEWAY_AZURE_BASE','https://api.ai.public.rakuten-it.com/azure-openai/global/v1')

print('key present:', bool(key))

# Try langchain_openai.OpenAIEmbeddings
try:
    from langchain_openai import OpenAIEmbeddings
    try:
        emb = OpenAIEmbeddings(model='text-embedding-3-small', base_url=base_openai, api_key=key)
        print('OpenAIEmbeddings object created, trying embed_documents...')
        v = emb.embed_documents(texts)
        print('OK, len', len(v))
    except Exception:
        print('OpenAIEmbeddings instantiation/usage failed')
        traceback.print_exc()
except Exception:
    print('import OpenAIEmbeddings failed')
    traceback.print_exc()

# Try AzureOpenAIEmbeddings
try:
    from langchain_openai import AzureOpenAIEmbeddings
    try:
        emb = AzureOpenAIEmbeddings(api_version='2023-05-15', azure_endpoint=base_azure, api_key=key)
        print('AzureOpenAIEmbeddings object created, trying embed_documents...')
        v = emb.embed_documents(texts)
        print('OK, len', len(v))
    except Exception:
        print('AzureOpenAIEmbeddings instantiation/usage failed')
        traceback.print_exc()
except Exception:
    print('import AzureOpenAIEmbeddings failed')
    traceback.print_exc()

# Try langchain.embeddings.OpenAIEmbeddings
try:
    from langchain.embeddings import OpenAIEmbeddings as LCOpenAIEmbeddings
    try:
        emb = LCOpenAIEmbeddings(openai_api_key=key, openai_api_base=base_openai, model='text-embedding-3-small')
        print('LC OpenAIEmbeddings object created, trying embed_documents...')
        v = emb.embed_documents(texts)
        print('OK, len', len(v))
    except Exception:
        print('LC OpenAIEmbeddings instantiation/usage failed')
        traceback.print_exc()
except Exception:
    print('import LC OpenAIEmbeddings failed')
    traceback.print_exc()

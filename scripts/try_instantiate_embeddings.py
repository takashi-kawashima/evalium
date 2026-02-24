import os
import traceback
from dotenv import load_dotenv

# load local.env to populate RAKUTEN_AI_GATEWAY_KEY
load_dotenv('local.env')

key = os.getenv('RAKUTEN_AI_GATEWAY_KEY')
base_openai = os.getenv('RAKUTEN_AI_GATEWAY_BASE','https://api.ai.public.rakuten-it.com/rakutenllms/v1/')
base_azure = os.getenv('RAKUTEN_AI_GATEWAY_AZURE_BASE','https://api.ai.public.rakuten-it.com/azure-openai/global/v1')

print('key present:', bool(key))
print('base_openai:', base_openai)
print('base_azure:', base_azure)

try:
    from langchain_openai import OpenAIEmbeddings
    try:
        e = OpenAIEmbeddings(model='text-embedding-3-small', base_url=base_openai, api_key=key)
        print('langchain_openai.OpenAIEmbeddings instantiated OK')
    except Exception:
        print('langchain_openai.OpenAIEmbeddings instantiation FAILED')
        traceback.print_exc()
except Exception:
    print('import langchain_openai.OpenAIEmbeddings failed')
    traceback.print_exc()

try:
    from langchain_openai import AzureOpenAIEmbeddings
    try:
        e = AzureOpenAIEmbeddings(api_version='2023-05-15', azure_endpoint=base_azure, api_key=key)
        print('langchain_openai.AzureOpenAIEmbeddings instantiated OK')
    except Exception:
        print('langchain_openai.AzureOpenAIEmbeddings instantiation FAILED')
        traceback.print_exc()
except Exception:
    print('import langchain_openai.AzureOpenAIEmbeddings failed')
    traceback.print_exc()

try:
    from langchain.embeddings import OpenAIEmbeddings as LCOpenAIEmbeddings
    try:
        e = LCOpenAIEmbeddings(openai_api_key=key, openai_api_base=base_openai, model='text-embedding-3-small')
        print('langchain.embeddings.OpenAIEmbeddings instantiated OK')
    except Exception:
        print('langchain.embeddings.OpenAIEmbeddings instantiation FAILED')
        traceback.print_exc()
except Exception:
    print('import langchain.embeddings.OpenAIEmbeddings failed')
    traceback.print_exc()

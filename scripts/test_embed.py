from evalium.evaluator import EmbeddingClient

client = EmbeddingClient()
try:
    v = client.embed_texts(["こんにちは"])
    print('shape', v.shape)
    print(v)
except Exception as e:
    import traceback
    traceback.print_exc()

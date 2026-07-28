import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession('har_model_fp32.onnx',
    providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
x = np.random.rand(1, 561).astype(np.float32)
for _ in range(20): sess.run(None, {'args_0:0': x})    # warmup (builds engine)
t0=time.time()
for _ in range(1000): sess.run(None, {'args_0:0': x})
print('GPU mean latency (ms):', (time.time()-t0)/1000*1000)

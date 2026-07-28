import onnxruntime as ort, numpy as np, time
s = ort.InferenceSession('har_model_fp32.onnx',
    providers=['CPUExecutionProvider'])
x = np.random.rand(1, 561).astype(np.float32)
for _ in range(10): s.run(None, {'args_0:0': x})   # warmup
t0 = time.time()
for _ in range(1000): s.run(None, {'args_0:0': x})
print('CPU mean latency (ms):', (time.time()-t0)/1000*1000)

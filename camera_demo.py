import cv2, numpy as np, time, os
import onnxruntime as ort

sess = ort.InferenceSession("mobilenetv2.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
print("Providers in use:", sess.get_providers())

labels = [l.strip() for l in open("imagenet_classes.txt")]
cap = cv2.VideoCapture(0)

os.makedirs("frames", exist_ok=True)
print("Running for ~60 seconds. Ctrl-C to stop early.")
start = time.time()
frame_count = 0

while True:
    t_frame = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break

    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = img.transpose(2, 0, 1)[None].astype(np.float32)

    t0 = time.perf_counter()
    out = sess.run(None, {"input": x})[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    top = int(out.argmax())
    end_to_end_ms = (time.perf_counter() - t_frame) * 1000
    fps = 1000.0 / end_to_end_ms

    cv2.putText(frame, f"{labels[top]}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"infer {infer_ms:.1f} ms | pipeline "
        f"{end_to_end_ms:.1f} ms | {fps:.1f} FPS", (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(f"frames/frame_{frame_count:05d}.jpg", frame)
    frame_count += 1

    if time.time() - start > 60:
        break

cap.release()
print(f"Done. {frame_count} frames saved to frames/")

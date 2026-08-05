#!/usr/bin/env python3
"""Benchmark PyTorch CUDA against ONNX Runtime CUDA using ResNet-18."""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torchvision.models as models

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Benchmark PyTorch CUDA against ONNX Runtime CUDA.")
    p.add_argument("--output-dir",type=Path,default=Path("results"))
    p.add_argument("--batch-sizes",type=int,nargs="+",default=[1,8,32])
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--iterations",type=int,default=50)
    p.add_argument("--seed",type=int,default=2026)
    return p.parse_args()

def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable.")

def export_model(model: torch.nn.Module,path: Path) -> None:
    x=torch.randn(1,3,224,224,device="cuda")
    torch.onnx.export(model,(x,),path,input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}},opset_version=18)
    onnx.checker.check_model(onnx.load(path))

def create_session(path: Path) -> ort.InferenceSession:
    preload=getattr(ort,"preload_dlls",None)
    if callable(preload): preload()
    s=ort.InferenceSession(str(path),providers=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"])
    if not s.get_providers() or s.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider is not primary: {s.get_providers()}")
    return s

def metadata(session: ort.InferenceSession,args: argparse.Namespace) -> dict[str,Any]:
    d=torch.cuda.get_device_properties(0)
    return {"python_version":sys.version,"platform":platform.platform(),"gpu_name":torch.cuda.get_device_name(0),"gpu_total_memory_bytes":int(d.total_memory),"gpu_compute_capability":f"{d.major}.{d.minor}","pytorch_version":torch.__version__,"pytorch_cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),"onnx_version":onnx.__version__,"onnxruntime_version":ort.__version__,"onnxruntime_available_providers":ort.get_available_providers(),"onnxruntime_active_providers":session.get_providers(),"batch_sizes":args.batch_sizes,"warmup_iterations":args.warmup,"benchmark_iterations":args.iterations,"random_seed":args.seed,"model":"torchvision.models.resnet18","weights":"ResNet18_Weights.DEFAULT","opset_version":18,"input_shape":"[batch_size, 3, 224, 224]","timing_note":"PyTorch uses CUDA events. ONNX Runtime uses synchronized wall-clock timing with GPU-resident input/output through I/O binding."}

def run(model,session,batch_sizes,warmup,iterations):
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name
    rows=[]
    print("Active providers:",session.get_providers())
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")
        x=torch.randn(batch_size,3,224,224,device="cuda",dtype=torch.float32).contiguous()
        with torch.inference_mode():
            for _ in range(warmup): model(x)
        torch.cuda.synchronize(); pt=[]
        with torch.inference_mode():
            for _ in range(iterations):
                a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
                a.record(); y_pt=model(x); b.record(); b.synchronize(); pt.append(float(a.elapsed_time(b)))
        io=session.io_binding()
        io.bind_input(name=input_name,device_type="cuda",device_id=0,element_type=np.float32,shape=tuple(x.shape),buffer_ptr=x.data_ptr())
        io.bind_output(name=output_name,device_type="cuda",device_id=0)
        for _ in range(warmup): session.run_with_iobinding(io)
        torch.cuda.synchronize(); ot=[]
        for _ in range(iterations):
            torch.cuda.synchronize(); t=time.perf_counter(); session.run_with_iobinding(io); torch.cuda.synchronize(); ot.append((time.perf_counter()-t)*1000)
        y_ort=io.copy_outputs_to_cpu()[0]; y_pt_np=y_pt.detach().cpu().numpy()
        pm=float(np.median(pt)); om=float(np.median(ot))
        rows.append({"batch_size":batch_size,"pytorch_median_ms":pm,"pytorch_p95_ms":float(np.percentile(pt,95)),"pytorch_throughput_images_per_second":batch_size*1000/pm,"onnx_median_ms":om,"onnx_p95_ms":float(np.percentile(ot,95)),"onnx_throughput_images_per_second":batch_size*1000/om,"onnx_speedup_vs_pytorch":pm/om,"onnx_latency_reduction_percent":(1-om/pm)*100,"max_absolute_error":float(np.max(np.abs(y_pt_np-y_ort)))})
    return pd.DataFrame(rows)

def main() -> int:
    args=parse_args(); require_cuda(); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model_path=args.output_dir/'resnet18_dynamic.onnx'; csv_path=args.output_dir/'onnx_runtime_cuda_benchmark.csv'; meta_path=args.output_dir/'onnx_runtime_environment.json'
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().to('cuda')
    export_model(model,model_path); session=create_session(model_path)
    df=run(model,session,args.batch_sizes,args.warmup,args.iterations); df.to_csv(csv_path,index=False)
    meta_path.write_text(json.dumps(metadata(session,args),indent=2)+'\n',encoding='utf-8')
    print(df.to_string(index=False)); print('Saved:',csv_path,meta_path)
    return 0
if __name__=='__main__': raise SystemExit(main())

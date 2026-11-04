"""Deterministic state digest preserving native tensor/NaN representation bytes."""
import hashlib
import json
import struct


def fingerprint(value):
    # Delayed scientific imports keep CPU admission fixtures model-free.
    import numpy as np
    import torch
    digest=hashlib.sha256()
    def token(kind,payload=b''):
        digest.update(kind.encode()+b':'+str(len(payload)).encode()+b':'+payload+b';')
    def visit(item):
        if isinstance(item,torch.Tensor):
            token('tensor-meta',json.dumps([str(item.dtype),list(item.shape)]).encode())
            token('tensor-bytes',item.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,np.ndarray):
            token('array-meta',json.dumps([item.dtype.str,list(item.shape)]).encode())
            token('array-bytes',np.ascontiguousarray(item).tobytes())
        elif isinstance(item,np.generic):
            token('numpy-scalar-meta',item.dtype.str.encode());token('numpy-scalar-bytes',item.tobytes())
        elif isinstance(item,dict):
            token('mapping-start')
            for key in sorted(item,key=lambda x:(type(x).__name__,repr(x))):visit(key);visit(item[key])
            token('mapping-end')
        elif isinstance(item,(tuple,list)):
            token(type(item).__name__+'-start')
            for child in item:visit(child)
            token(type(item).__name__+'-end')
        elif isinstance(item,float):token('float64-bytes',struct.pack('<d',item))
        elif item is None:token('none')
        elif isinstance(item,bool):token('bool',str(item).encode())
        elif isinstance(item,int):token('integer',str(item).encode())
        elif isinstance(item,str):token('string',item.encode())
        else:raise TypeError('Unsupported native state fingerprint member: '+type(item).__name__)
    visit(value)
    return digest.hexdigest()

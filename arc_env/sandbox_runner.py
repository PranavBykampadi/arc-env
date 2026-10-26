"""Runs a candidate transform() on a list of input grids, in a separate process. Reads JSON on stdin:
{"code": "...", "inputs": [[...], ...]} and prints JSON {"outputs": [grid or null], "errors": [str or null]}."""
import json, sys, resource, signal

def main():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    except Exception:
        pass
    req = json.load(sys.stdin)
    ns = {}
    outs, errs = [], []
    try:
        exec(compile(req["code"], "<transform>", "exec"), ns)
        fn = ns.get("transform")
        if not callable(fn):
            raise RuntimeError("no transform(grid) function defined")
    except BaseException as e:
        print(json.dumps({"outputs": [None] * len(req["inputs"]), "errors": [f"{type(e).__name__}: {e}"] * len(req["inputs"])}))
        return
    for g in req["inputs"]:
        def alarm(*_):
            raise TimeoutError("transform() took too long")
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(3)
        try:
            out = fn([row[:] for row in g])
            try:
                import numpy as np
                if isinstance(out, np.ndarray):
                    out = out.tolist()
            except ImportError:
                pass
            out = [[int(v) for v in row] for row in out]
            if not out or any(len(r) != len(out[0]) for r in out):
                raise ValueError("ragged or empty output")
            outs.append(out); errs.append(None)
        except BaseException as e:
            outs.append(None); errs.append(f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            signal.alarm(0)
    print(json.dumps({"outputs": outs, "errors": errs}))

if __name__ == "__main__":
    main()

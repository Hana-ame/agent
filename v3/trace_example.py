import trace
import io

def foo():
    bar()

def bar():
    pass

tracer = trace.Trace(count=0, trace=1)
out = io.StringIO()
tracer.runfunc(foo)

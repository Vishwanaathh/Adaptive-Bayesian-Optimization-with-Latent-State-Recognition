import numpy as np
from abo_lsr.space import SearchSpace, Dimension

def test_encode_decode_roundtrip():
    s=SearchSpace([Dimension("lr","log",.001,.1),
                   Dimension("depth","integer",2,10),
                   Dimension("kind","categorical",choices=("a","b","c"))])
    p={"lr":.01,"depth":6,"kind":"b"}
    x=s.encode(p); q=s.decode(x)
    assert abs(q["lr"]-p["lr"])<1e-9
    assert q["depth"]==p["depth"]
    assert q["kind"]==p["kind"]

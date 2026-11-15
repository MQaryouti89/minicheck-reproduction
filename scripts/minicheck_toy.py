"""Toy check that MiniCheck checkers load and give sensible scores (no gated data needed)."""
import sys, time
from minicheck.minicheck import MiniCheck
doc = ("The Prophet's Mosque in Medina was built in 622 CE. It was later expanded several times, "
       "and today it can hold more than one million worshippers.")
claims = ["The Prophet's Mosque was built in 622 CE.",          # supported
          "The Prophet's Mosque was built in 700 CE.",          # contradicted
          "The mosque can hold over a million worshippers.",    # supported
          "The mosque was designed by a French architect."]     # unsupported
for name in sys.argv[1:]:
    t = time.time()
    m = MiniCheck(model_name=name, cache_dir="ckpts")
    labels, probs, _, _ = m.score(docs=[doc] * len(claims), claims=claims)
    print(f"{name:18} load+score {time.time()-t:5.1f}s  labels={labels}  probs={[round(p,3) for p in probs]}", flush=True)

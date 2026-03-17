import json
import codecs
from pluto.stages.verify import _parse_verify

text = codecs.open('verify_dump.txt', 'r', 'utf-8').read()
out = _parse_verify(text)

print(f"Parsed Checked Claims Count: len(out.verification.checked_claims)")
print(f"Total: {len(out.verification.checked_claims)}")
supported = [cc for cc in out.verification.checked_claims if cc.status.value == "supported"]
print(f"Supported: {len(supported)}")

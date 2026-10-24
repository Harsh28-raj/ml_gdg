# Sample images
Real photos from Wikimedia Commons (CC licensed, attribution in [images/LICENSES.md](images/LICENSES.md)).

| File | Content | Used for | Status |
|---|---|---|---|
| garbage1.jpg | dumped waste on a street (London) | classify → waste | ✅ |
| garbage2_a.jpg / garbage2_b.jpg | same garbage spot, two angles | duplicate → true; verify same spot + garbage still there → Suspicious | ✅ |
| garbage3.jpg | roadside garbage dump (Mumbai) | classify → waste; different spot vs garbage1 → not duplicate | ✅ |
| pothole1.jpg | pothole (Bengaluru) | classify → road, duplicate, verify | ✅ |
| pothole1_b.jpg | same pothole, second shot | duplicate → true; verify re-shot unfixed → Suspicious | ✅ |
| water1.jpg / water2.jpg | waterlogged street / leaking pipe | classify → water | ✅ |
| streetlight1.jpg | lamp post on a waterfront | weak sample (classifies `other`) — replace with a clearly broken streetlight | ⚠️ |
| clean_road_other.jpg | clean highway, different place | verify → Suspicious | ✅ |
| **pothole1_fixed.jpg** | **same spot as pothole1, repaired** | verify → Verified | ❌ needs your photo |
| **garbage1_clean.jpg** | **same spot as garbage1, cleaned** | verify → Verified | ❌ needs your photo |

Verified needs a real before/after pair of the **same spot**. Easiest: take two phone photos of a littered spot before and after cleaning it, save as `garbage1.jpg` + `garbage1_clean.jpg` (replace garbage1) or add your own pair and pass both to `/ml/verify`. Then run `python scripts/tune.py` and `pytest tests/test_verify.py`.

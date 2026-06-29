"""Model-free n-gram drafter: prefix (last k tokens) -> most-frequent continuation."""
from collections import Counter, defaultdict

from base import Drafter  # flat import: run self-check from this dir (matches workload/)


class NgramDrafter(Drafter):
    name = "ngram"
    model_free = True

    def __init__(self, prefix_len: int = 3):
        self.prefix_len = prefix_len
        self.store: dict[tuple, Counter] = defaultdict(Counter)

    def build_datastore(self, token) -> bool:
        n = self.prefix_len
        for i in range(len(token) - n):
            self.store[tuple(token[i:i + n])][token[i + n]] += 1
        return len(self.store) > 0

    def get_draft_token(self, prompt_token, n_token, user_id=None, hidden_state=None):
        # ponytail: greedy single path; tree/multi-candidate draft is YAGNI until measured
        seq, prefix = [], list(prompt_token[-self.prefix_len:])
        for _ in range(n_token):
            cont = self.store.get(tuple(prefix))
            if not cont:
                break  # unseen prefix -> stop drafting
            nxt = cont.most_common(1)[0][0]
            seq.append(nxt)
            prefix = prefix[1:] + [nxt]
        return [seq]

    def describe_datastore(self):
        peaks = sorted((c.most_common(1)[0][1] for c in self.store.values()), reverse=True)
        return peaks, len(self.store)


if __name__ == "__main__":
    d = NgramDrafter(prefix_len=2)
    assert d.build_datastore([1, 2, 3, 1, 2, 3, 1, 2, 3])
    # (1,2)->3, (2,3)->1, (3,1)->2 ; draft follows the cycle
    assert d.get_draft_token([0, 1, 2], 3) == [[3, 1, 2]]
    assert d.get_draft_token([9, 9], 2) == [[]]          # unseen prefix -> empty
    assert d.get_draft_token([0, 2, 3], 1) == [[1]]       # prefix (2,3)->1
    peaks, total = d.describe_datastore()
    assert total == 3 and peaks[0] == 3                   # (1,2)->3 seen 3x
    print("ngram drafter OK", d.describe_datastore())

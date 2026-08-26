# FinHRM QP3-CORE S4 safe replay

- Gate: **PASS**
- Eligible SELECT/ADD: `363` (`{'ADD': 123, 'SELECT': 240}`)
- Gold actions: `{'ABSTAIN': 92, 'ANSWER': 257, 'REJECT': 14}`; execution-correct gold: `363/363`
- Permutation exact: `363/363`
- Explicit counterfactuals: `141`; false accepts: `0`
- Company groups: `121`; macro/worst gold answer rate: `0.515/0.000`
- Layout groups: `6`; macro/worst gold answer rate: `0.582/0.000`
- No ABW, validation/blind/test data, threshold/checkpoint change, fine-tuning, or Phase-D unlock.
- This is development/train stress evidence, not unseen or universal report coverage.

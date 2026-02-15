Which Feature Engineering Approach is best with NN's?
I've performed Feature Engineering which works good with Boosting Models, but not as well with NN's like RealMLP, TabM etc. I want to know further Feature Engineering Techniques that Specifically work good with NN's

Standard FE for GBDTs, like interactions/ratios/conditions, are mostly useless for NNs here. Even non-linear transformations like polynomials/log/exp/sqrt sometimes give nothing at all on these synthetic datasets. The most crucial thing is that NNs are very sensitive to changing in the feature landscape, so adding new columns can potentially harm performance unless parameters are being adjusted accordingly. Accumulated experience from past Playground episodes included following techniques that work best with NNs:

Convert initial numericals (some or all) into categoricals and stack them with raw numericals.
Perform cyclic encoding (sin/cos features on raw nums) with carefully tuned periodic values (TabM can do that internally by periodic encoding).
Selectively create interaction categories (aka string combos) from init nums + cats feats.
Perform target/count encoding on high cardinality features, especially on combos (encoded feats stay unscaled before passing to NN).
Optionally try discretizing raw numericals as well as newly created target/count encoded (TabM can do that internally by piecewise linear encoding).
Ordinal encoding (with tuned ordinal values) might work if raw nums have ordinal categories.
Digit extraction and rounding from raw nums with a carefully tuned depth level, then applying categorization from the first point.
Very important is relying on CV/LB improvement in the first place during engineering/selection rather than various feature importances.

Reply

1
Tilii
Posted 12 hours ago

Great summary. One point where I slightly disagree is this:

Perform target/count encoding on high cardinality features, especially on combos (encoded feats stay unscaled before passing to NN).

In my hands NNs generally handle high-cardinality features better than GBMs. It is true that encoding or feature discretization will help all the methods, but it is less needed by NNs.

Reply

React
Vladimir Demidov
Posted 6 hours ago

Alright, maybe I should have mentioned that selection is key here as well, instead of appealing to cardinality, because applying encoding to all features with the hope that each will contribute is quite risky.

Reply

React

4 more replies
Profile picture for Tilii
Profile picture for Muhammad Hamza
Profile picture for Vladimir Demidov
Profile picture for Mahog
Qudsiya Siddique
Posted a day ago

· 345th in this Competition

A nueral network for tabular dataset should be noo, work with boosting models , try stacking oof predictions. , as for feature engineering a minimal ratios of some columns are good but dont create too many features , this will create noise in your dataset

Reply

React

I’m a bit concerned about the trend of increasing submission limits in competitions. This one allows 10 submissions per day, which feels excessive.

In theory, more submissions should encourage experimentation. In practice, it mostly incentivizes blind leaderboard probing. With enough attempts, you can keep nudging blend weights, combining random checkpoints, or even mixing loosely related models until the public LB ticks up.

Even with a 5-submission limit, the previous competition ended up with the top ~30 notebooks (sorted by public score) being almost entirely blind blending. No insights, no ablations; just “here’s a pile of models and some weights that happened to work on the public LB.”

What’s especially telling is that even people who published genuinely insightful, well-explained notebooks eventually felt compelled to release blind blend versions as well, often keeping their original, non-blending notebook pinned while adding a separate “blend-for-LB” notebook on top. That kind of behavior feels less like curiosity and more like adapting to a broken incentive structure.

At this point, higher submission limits seem to reward persistence and noise exploitation more than actual understanding, and they make the public leaderboard an even worse proxy for solution quality.

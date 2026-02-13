## Hi everyone,

This is my first Kaggle competition, so I wanted to share a quick reflection on my process and what I’m learning. I reached a strong public LB rank (rank #13 — tied for rank #7) using a blind blend of different models. It worked, but I now see why that can be a mistake — and why I shouldn’t treat a high public score as the end goal.

## What I did:

Found several strong, diverse models.
Sanity‑checked that IDs aligned.
Used a conservative blend (not fancy, just stable).
Iterated a few times and stopped once performance plateaued.
Why it “worked” (and why that can be misleading)
Blind blending can look great on the public leaderboard because it can accidentally fit the public split. But the public LB is only a slice of the data — so a blend can “look” better without actually generalizing. With high submission limits, it’s easy to tune to the public LB by accident and confuse that with real progress.

## What I learned:

A high public score doesn’t necessarily mean a better model.
Blind blending can be a trap if you’re not validating properly.
The real goal should be learning and building methods that generalize.

## What I plan to do next:

Focus on OOF‑based ensembling or stacking instead of blind weights.
Track cross‑validation metrics more seriously.
Spend more time on understanding model diversity instead of just “higher LB.”
I’m new to Kaggle and still learning, and I’d really appreciate perspectives from more experienced Kagglers. If you have advice on improving ensembling practices (or avoiding common pitfalls), I’m all ears.

Thanks for reading — happy to connect and learn together.

@tilii7 (thank you for your help!)

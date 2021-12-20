# Are You The One CP Analysis

Uses integer constraint programming to model the MTV game show _Are You The One_. The game is played by 10 men and 
10 women (until recently). There is a perfect bipartite matching between men and women determined beforehand by the 
show's matchmakers.  In each round of the game, there are three stages:
1. there's a contest, e.g. trivia, the winners of which are a set of couples
2. people not in those winning couples vote on a couple to send to the _Truth Booth_, which reveals whether that couple
    is in the perfect matching
3. people form couples (any matching is allowed) and the host informs them of how many correct matches they got (but
    not which couples are correct)

We can model this using integer constraint programming to encode the constraints imposed by learning the outcome of the
Truth Booth or the matching ceremony.  See the [notebook](https://github.com/theodore-ando/areyoutheone-analysis/blob/master/AreYouTheOneAnalysis.ipynb) 
for more analysis (markdown won't render math).

See [areyoutheone.py](https://github.com/theodore-ando/areyoutheone-analysis/blob/master/areyoutheone.py) for the main 
class you can interact with.
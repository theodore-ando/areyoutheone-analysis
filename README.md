# Are You The One CP Analysis

Uses integer constraint programming to model the MTV game show _Are You The One_. The game is played by 10 men and 
10 women (until recently). There is a perfect bipartite matching between men and women determined beforehand by the 
show's matchmakers.  In each round of the game, there are three stages:
1. there's a contest, e.g. trivia, the winners of which are a set of couples
2. people not in those winning couples vote on a couple to send to the _Truth Booth_, which reveals whether that couple
    is in the perfect matching
3. people form couples (any matching is allowed) and the host informs them of how many correct matches they got (but
    not which couples are correct)

We model this by the following constraint problem.  Let $i, j \in [1,10]$.  Let $x_{i,j}$ be the $[0,1]$ variable 
representing whether man $i$ and woman $j$ are matched together.  Then:
$$\sum_{j=1}^{10} x_{i,j} = 1 \qquad \forall i \in [1,10]$$
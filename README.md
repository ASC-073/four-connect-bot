# four-connect-bot

**Acknowledgements**: Thanking Dr. Sujith Thomas and the Teaching Assistants for Artificial Intelligence (CS F407) course for designing and guiding in the completion of the assignment.

The project was part of the course Artificial Intelligence (CS F407), entirely using Python. Various evaluation functions were tried out to minimise the number of moves taken by the bot to win and win rate. Move ordering heuristic was implemented along with Alpha-Beta pruning to improve efficiency.

**Problem Statement**: Given a myopic opponent player’s code and a test case ongoing game, create a bot that succeeds the test case using evaluation functions, move ordering heuristics and alpha-beta pruning.


**Metrics**: Number of times won out of 50, and average moves taken to win.


**EVALUATION FUNCTIONS**:

	(i) Most simple evaluation function used was counting game tree player’s pieces (score) occuring in same rows, columns and diagonals and subtracting the opponent’s “score” then returning it, without attaching any kind of weights to each combination.

	(ii) Next, gave linear weights to each kind of combination (4 or more in a row 	got weight – 4, 3 in a row – 3, 2 in a row – 2, 1 individual – 1).

	(iii) Next, gave exponential weights (4 or more in a row – 100000, 3 in a row – 1000, 2 in a row – 100, 1 individual – 10).

	(iv) Tried another evaluation function which prioritised center positions and gives more weight to more number of player’s two-coin configurations, but (iii) is most suitable for our needs.

Evaluation function (iii) is optimal for our needs, and we even see that we win almost all of the games upon increasing the cutoff depth to 5 and the average moves to win have reduced significantly especially compared to (i) and (ii).

**TESTBENCH**: Amended the PlayGame() function for win rate, average moves performance.

**TESTCASE**: is passing in 3 moves using the evaluation function (iii).

Implemented alpha-beta pruning as well, but since it only affects time taken by the algorithm and not wins, didn’t include it in the table.

**Move Ordering Heuristic**: Using a simple move ordering such that the bot prefers putting a coin somewhere in the center than the corner (i.e. most weight to center, depreciating weight linearly as we go to corners) improves the time taken by bot to win against the myopic player.

**Scope of Improvement**:

	(i) Increasing reusability using Object Oriented Programming and Modularity
 	(ii) Further optimal evaluation function and move ordering heuristics to decrease number of moves to win and increase win rate.
  	(iii) Introducing reinforcement learning to enhance the bot's optimisation of strategies and exploration of game strategies.

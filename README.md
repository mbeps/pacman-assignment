## 6CCS3AIN Coursework

### 1. Introduction
This coursework exercise asks you to write code to create an **MDP-solver** to work in the Pacman environment that we used for the practical exercises.

> **Note**: Read all these instructions before starting.  
> This exercise will be assessed.

---

### 2. Getting Started
- Download the file `pacman-cw.zip` from KEATS.
- It contains:
  - A familiar set of files that implement Pacman.
  - Version 6 of `api.py`, defining the environment's observability and using the same non-deterministic motion model as the practicals.
- **What’s new in `api.py` (Version 6)**:
  - Pacman can now see:
    - All objects (walls, food, capsules, ghosts).
    - The **state of ghosts** (e.g., edible or not).

---

### 3. What You Need to Do

#### 3.1 Write Code
Develop a program to control Pacman and win games using an **MDP-solver**. The model must include:

- **States** (S): A finite set.
- **Actions** (A): A finite set.
- **State-transition function**: $P(s'|s,a)$
- **Reward function**: $R$
- **Discount factor**: $\gamma \in [0,1]$

You must compute actions using one of the following methods:
- **Value Iteration**
- **Policy Iteration**
- **Modified Policy Iteration**

**Goals:**
1. Win in `smallGrid`.
2. Win in `mediumClassic`.

> **Note**: Winning means eating all the food and completing the game successfully. Score is irrelevant for passing this section.

---

#### 3.1.1 Getting Excellence Points
Earn up to **20 additional points** by achieving a **high Excellence Score Difference ($\Delta S_e$)** in the `mediumClassic` layout. This measures **average winning score**.

- $\Delta S_e = \sum_{i \in W} (s_w(i) - 1500)$, where:
  - $W$: Set of games won.
  - $s_w(i)$: Score in game $i$.

> **Notes:**
> - Losses are ignored ($\Delta S_e = 0$ if negative).  
> - Excellence points depend on high scores in `mediumClassic` only.

---

### 3.2 Things to Bear in Mind

#### Key Commands
1. **Run in `smallGrid` (25 games):**
   ```bash
   python pacman.py -q -n 25 -p MDPAgent -l smallGrid
   ```

2. **Run in `mediumClassic` (25 games):**
   ```bash
   python pacman.py -q -n 25 -p MDPAgent -l mediumClassic
   ```

#### Evaluation Constraints
- Time limits:
  - **`smallGrid`**: 5 minutes.
  - **`mediumClassic`**: 25 minutes.
- Runs on a high-performance machine (26 cores, 192 GB RAM).
- Same agent instance is used across all 25 games.
- Use the `final()` function to reset state variables.

---

#### Additional Notes
- Only **Python 2.7** libraries are allowed.  
- Code must follow a consistent style and include clear comments.  
- Use provided MDP methods. **Q-learning is not allowed.**

---

### 3.3 Limitations
1. **Language**: Code must be in **Python 2.7**.  
   - Python 3 or other languages will not be marked.
2. **Interaction**: Use only functions in **`api.py (Version 6)`**.
3. **File Modifications**: You may only modify `mdpAgents.py`.
4. **Plagiarism**: Acknowledge borrowed code with comments.
5. **MDP-Specific**: Solutions must be based solely on MDP-solving methods.


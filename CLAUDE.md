# CLAUDE.md - Poker Bot Project Context

Rules for Claude:
1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md. Specifically, read mccfr_research_comprehensive.md for researched information. This is very important when you are implementing technical features.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

The todo.md file contains what you have done and what you should do.

## Project Overview

This is a world-class poker bot implementation for 6-player Texas Hold'em No-Limit using **Monte Carlo Counterfactual Regret Minimization (MCCFR)** as the core algorithm. The project follows a principled three-phase approach: first achieving near-zero exploitability with MCCFR, then enhancing the strategy with advanced techniques including neural approximation and reinforcement learning for dynamic abstraction optimization.

### High-Level Architecture

1. **Phase 1: Core MCCFR Implementation** - Build theoretically sound MCCFR with external sampling to achieve ≤1 mbb/hand exploitability
2. **Phase 2: Neural Network Integration** - Implement Deep CFR for scalable strategy representation and faster inference
3. **Phase 3: Advanced Optimization** - Apply RL-CFR for dynamic abstraction learning and continuous betting size optimization
4. **Phase 4: Tournament Evaluation** - Comprehensive testing against baselines and exploitability measurement

## Key Technical Components

### Environment Setup
- **Game**: 6-player No-Limit Texas Hold'em
- **Framework**: PettingZoo `texas_holdem_no_limit_v6` environment configured for 6 players
- **Initial Action Abstraction**: Fixed bet sizes (Check/Call, 1/4 pot, 1/2 pot, 1x pot, All-In)
- **Card Abstraction**: Strength-based hand bucketing with suit isomorphism
- **Observation Space**: Cards, chip stacks, betting history, and position information

### Core MCCFR Implementation
- **Algorithm**: External-sampling Monte Carlo CFR for optimal balance of efficiency and convergence
- **Sampling Strategy**: Sample opponent and chance actions, traverse all own actions
- **Regret Storage**: Tabular regret tables with information set hashing
- **Strategy Computation**: Regret matching with positive regret normalization
- **Convergence Target**: ≤1 mbb/hand exploitability (≤0.1 mbb/hand for professional level)

### Information Set Representation
- **Card Encoding**: Efficient indexing of hole cards and community cards
- **Action Abstraction**: Discrete betting buckets to limit action space
- **History Compression**: Compact representation of betting sequences
- **Position Awareness**: Incorporate seat position and relative stack sizes

### Memory Management
- **Regret Tables**: Hash-based storage for information set regrets
- **Strategy Summation**: Accumulate strategy weights for average policy computation
- **Reservoir Sampling**: O(1) space complexity buffer for experience replay (later phases)
- **Parallelization**: Multi-threaded regret updates with lock-free data structures

### Neural Network Integration (Phase 2)
- **Deep CFR**: Neural networks replace tabular regret storage for scalability
- **Architecture**: Transformer-based information set encoder
- **Training**: Supervised learning on MCCFR-generated (infoset, regret) pairs
- **Inference**: Fast strategy lookup without explicit abstraction

### Advanced Optimization (Phase 3)
- **RL-CFR**: Reinforcement learning for dynamic action abstraction selection
- **Continuous Betting**: Real-time subgame solving for optimal bet sizing
- **Opponent Modeling**: Bayesian learning of opponent archetypes and patterns
- **Exploitation Modules**: Safe deviation from GTO based on opponent weaknesses

## Project Structure

```
poker_bot/
├── envs/
│   └── holdem_wrapper.py          # 6-player PettingZoo environment wrapper
├── core/
│   ├── card_utils.py              # Card encoding and hand evaluation
│   ├── info_set.py                # Information set representation
│   └── regret_storage.py          # Regret table management
├── solvers/
│   ├── mccfr_trainer.py           # Core external-sampling MCCFR
│   ├── deep_cfr_trainer.py        # Neural network CFR approximation
│   └── rl_cfr_trainer.py          # RL-based dynamic abstraction
├── neural/
│   ├── info_set_encoder.py        # Transformer for information sets
│   └── opponent_modeling.py       # Neural opponent pattern recognition
├── agents/
│   ├── baseline_agent.py          # MCCFR strategy agent
│   ├── neural_agent.py            # Deep CFR agent
│   └── adaptive_agent.py          # RL-enhanced agent
├── evaluation/
│   ├── evaluator.py               # Exploitability measurement
│   ├── tournament_runner.py       # Multi-agent tournaments
│   └── benchmark_agents.py        # Test opponents
└── utils/
    ├── logger.py                  # Logging and monitoring
    ├── memory.py                  # Reservoir sampling buffer
    └── analysis_tools.py          # Strategy visualization

scripts/
├── 01_train_mccfr.py             # Train core MCCFR blueprint
├── 02_eval_mccfr.py              # Evaluate MCCFR exploitability  
├── 03_train_deep_cfr.py          # Neural network training
├── 04_train_rl_cfr.py            # RL abstraction optimization
└── 05_tournament.py              # Final evaluation tournament
```

## Implementation Order

### Phase 1: Core MCCFR Foundation
1. **holdem_wrapper.py** - 6-player PettingZoo environment wrapper with action masking
2. **card_utils.py** - Card encoding, hand evaluation, and abstraction utilities  
3. **info_set.py** - Information set representation and hashing
4. **mccfr_trainer.py** - External-sampling MCCFR algorithm implementation
5. **regret_storage.py** - Efficient regret table management and strategy computation
6. **evaluator.py** - Exploitability measurement and best response calculation
7. **baseline_agent.py** - MCCFR strategy wrapper for gameplay

### Phase 2: Neural Network Integration  
8. **deep_cfr_trainer.py** - Neural network approximation of regret values
9. **info_set_encoder.py** - Transformer architecture for information set encoding
10. **neural_agent.py** - Deep CFR strategy agent with fast inference

### Phase 3: Advanced Optimization
11. **rl_cfr_trainer.py** - Reinforcement learning for dynamic abstraction
12. **continuous_betting.py** - Real-time subgame solving for bet sizing
13. **opponent_modeling.py** - Adaptive strategies and exploitation modules

### Phase 4: Evaluation and Testing
14. **tournament_runner.py** - Multi-agent evaluation framework
15. **benchmark_agents.py** - Baseline opponents and test scenarios
16. **analysis_tools.py** - Strategy analysis and visualization utilities

## Key Research Findings

### MCCFR Algorithm Insights
- **External sampling** provides optimal balance: samples opponent/chance actions, traverses all own actions
- **Convergence rate**: O(1/√T) with theoretical guarantees for Nash equilibrium
- **Exploitability targets**: ≤1 mbb/hand is competitive, ≤0.1 mbb/hand is professional level
- **Variance considerations**: External sampling has lower variance than outcome sampling
- **Scalability**: Dramatically reduces per-iteration cost compared to vanilla CFR

### Sampling Technique Comparison
- **Chance Sampling**: Good starting point, samples card deals only
- **Outcome Sampling**: Best for huge action spaces, highest variance
- **External Sampling**: Industry standard for poker, optimal efficiency/accuracy tradeoff
- **Implementation**: External sampling easier to parallelize and debug

### Deep CFR Breakthroughs
- **Abstraction elimination**: Neural networks replace manual abstraction design
- **Generalization**: Networks learn strategic patterns across similar states
- **Scalability**: Handles full no-limit games without explicit abstraction
- **Performance**: Achieves superhuman play in heads-up no-limit hold'em
- **Memory efficiency**: 100-1000× faster inference than tabular storage

### RL-CFR for Dynamic Abstraction
- **Problem**: Fixed betting abstractions lose strategic nuance
- **Solution**: RL learns situation-specific action abstractions
- **Performance**: Substantial improvement over fixed abstractions in HUNL
- **Efficiency**: No increase in CFR solving time
- **Adaptability**: Abstractions adapt to different game situations automatically

### Continuous Betting Size Solutions
- **DeepStack approach**: Real-time subgame solving with neural value estimates
- **RL-CFR approach**: Dynamic abstraction selection based on game state
- **Gradient optimization**: Treat bet sizing as continuous optimization problem
- **Action clustering**: Learn strategic equivalence of different bet sizes

### Multi-Agent Considerations
- **6-player complexity**: Exponentially larger strategy space than heads-up
- **Coalition dynamics**: Players may form temporary alliances
- **Position importance**: Seat position critically affects optimal strategy
- **Stack size effects**: Relative chip stacks change optimal play significantly



# Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

## File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the
  gemini command:

### Examples (some example files are not actually in this project):

**Single file analysis:**
gemini -p "@src/main.py Explain this file's purpose and structure"

Multiple files:
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

Entire directory:
gemini -p "@src/ Summarize the architecture of this codebase"

Multiple directories:
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

Current directory and subdirectories:
gemini -p "@./ Give me an overview of this entire project"

# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"

Implementation Verification Examples

Check if a feature is implemented:
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

Verify authentication implementation:
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

Check for specific patterns:
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

Verify error handling:
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

Check for rate limiting:
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

Verify caching strategy:
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

Check for specific security measures:
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

Verify test coverage for features:
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

When to Use Gemini CLI

Use gemini -p when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

Important Notes

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results
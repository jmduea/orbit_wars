# Brain Dump

Centralized repository of all the things that i've felt the need to write down in relation to this project ranging from ideas to questions and issues that need to be addressed.

## Ideas

- Add informative wandb.tags for config groups that:

    - override tags for their config group 
        
        - ex: model/base.yaml defines `wandb.tags` that quickly identify key model architecture choices. model/*.yaml the tag(s) for the specific config values that were overridden.

    - append to tags of other config groups (i.e. model/*.yaml doesn't override tags of task/*.yaml, but appends to them)

    - allows for quick filtering of runs by config group in wandb UI

- Seperate num_envs being defined directly by format:

    - format/*.yaml defines num_envs per 2p/4p grouping with both scalar values and percentage weighting values
    - unclear how num_envs is determined for each rollout group
    - **potentially worth removing format/*.yaml and instead defining num_envs (total) & 2p/4p weighting values directly in training/*.yaml, and calculating num_envs for each rollout group based on the weighting values and total num_envs.**

    - Rough example:
    ```yaml
    training:
        num_envs: 64
        2p_weight: 0.8
        4p_weight: 0.2
        # calculated num_envs
        # 2p_num_envs = floor(num_envs * 2p_weight)
        # 4p_num_envs = floor(num_envs * 4p_weight)
        # would need to adress how we handle cases where the total num_envs is not a multiple of the number of players (i.e. 64 * 0.8 = 51.2)
    ```

- Update terminal_reward mode from binary_win to normalized ship differential

    - binary_win is a pretty sparse reward, and also doesn't completely capture how kaggle handles determining the winner.

        - From kaggle docs:
        ```
        Scoring and Termination
        The game ends when:
        - Step limit reached: 500 turns.
        - Elimination: Only one player (or zero) remains with any planets or fleets.
        - Final score = total ships on owned planets + total ships in owned fleets. Highest score wins.
        ```

    - Ideally, we take the normalized ship differential and scale it to be between -1 and 1.
        - 1 signifies winning by elimination
        - -1 signifies losing by elimination
        - 0 signifies an ABSOLUTE tie, i.e. the exact same number of ships on owned planets and owned fleets (HIGHLY UNLIKELY)
        - anything from -.99->-0.01 represents a loss, with severity of loss increasing as the value approaches -1
        - anything from 0.01->0.99 represents a win, with severity of win increasing as the value approaches 1

    - My hunch is that this reward scale is more interpretable for agents than the current binary_win reward.

- Implement tournament system, ranking based on official ranking system and use tourney evals locally to evaluate the performance of the best performing agents.

- Build VRAM profile for configuration parameters based on wandb run data/reports:
    
    - avg compile time (time to third update)
    - avg VRAM usage, average RAM usage
    - u30 wall time/average u wall time/average game (500 steps) wall time
    - 1m step avg wall time
    - avgs w/compile time factored out

- Add a debug metric that tracks the average number of ships per fleet launched (overall, by learner), idea is to help identify if the agent is learning "proper" ship sizing for fleet launches.

## Questions

- What's the main cause of long compile time for training runs? Jax is known to be slow to compile, but not sure if the compile time of the average training run is within expected bounds.
    
    - potentially worth escalating to an issue if it's not within expected bounds.

- What exactly is survival_time? How is it calculated? How does it relate to agent performance?

## Issues

- Submissions generated through the artifact pipeline are not passing validation when submitted to Kaggle.
    
    - submission used: 
    ```
    /home/jmduea/projects/orbit_wars/outputs/campaigns/default/runs/20260526T174954Z-s42-f83967b0/evaluations/replay_u000100_0dfe6ddb45ef4b6ba98b90d9973bca68/docker_validation/submission.tar.gz
    ```
    - error log files:
    ```
    /home/jmduea/projects/orbit_wars/outputs/campaigns/default/runs/20260526T174954Z-s42-f83967b0/logs/78036470.json

    /home/jmduea/projects/orbit_wars/outputs/campaigns/default/runs/20260526T174954Z-s42-f83967b0/logs/78036470-0.json

    /home/jmduea/projects/orbit_wars/outputs/campaigns/default/runs/20260526T174954Z-s42-f83967b0/logs/78036470-1.json
    ```

- Current kaggle population worker pipeline is broken and needs to be fixed. The kernel gets uploaded to kaggle, but then fails to run, fails to pick up wandb secret as well. Manual workarounds work to make the worker run by importing the kaggle notebook into colab and running the worker manually. No manual workaround currently works to get the worker to run in kaggle.

- Current config setup is a bit messy and needs to be cleaned up.

    - There are a lot of config groups that are not being used and could be removed.
    - There are seemingly redundant/unused config values that could be removed.
    - Better documentation of config setup and usage is needed. Especially around what each value does and how it affects the training run.

- Debug seed swapping during training. The seed scheduler should be swapping seeds periodically to help with training stability. Evidence of whether this is actually happening is needed.